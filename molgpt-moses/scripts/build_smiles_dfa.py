#!/usr/bin/env python3
"""Compile and validate the empirical structural token-DFA for MOSES SMILES."""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import moses


CAP = 8
BUILD_FRACTION = 0.9
VOCAB_SIZE = 30
BOS_ID = 1
EOS_ID = 2
ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "data" / "tokenizer_content.json"
DFA_PATH = ROOT / "grammars" / "smiles_structural.token_dfa.json"
REPORT_PATH = ROOT / "grammars" / "dfa_build_report.json"

OUTSIDE, B0, BN, BH = "OUTSIDE", "B0", "Bn", "BH"


def load_vocab() -> dict[str, int]:
    with TOKENIZER_PATH.open() as f:
        tokenizer = json.load(f)
    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) == VOCAB_SIZE, len(vocab)
    assert vocab["[BOS]"] == BOS_ID and vocab["[EOS]"] == EOS_ID
    return vocab


def corpus_stats(smiles_list: list[str]):
    bigram: dict[str, set[str]] = defaultdict(set)
    pre_eos: set[str] = set()
    max_depth = 0
    max_open_rings = 0
    for smiles in smiles_list:
        assert smiles, "MOSES corpus unexpectedly contains an empty SMILES"
        prev = "BOS"
        depth = 0
        ringmask = 0
        for char in smiles:
            bigram[prev].add(char)
            prev = char
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth -= 1
                assert depth >= 0, smiles
            elif char in "123456":
                ringmask ^= 1 << (int(char) - 1)
                max_open_rings = max(max_open_rings, ringmask.bit_count())
        assert depth == 0 and ringmask == 0, smiles
        pre_eos.add(smiles[-1])
    return bigram, pre_eos, max_depth, max_open_rings


def state_name(state: tuple[str, int, int, str]) -> str:
    prev, depth, ringmask, bstage = state
    return f"s|{prev}|{depth}|{ringmask}|{bstage}"


def next_state(state, char: str, bigram: dict[str, set[str]]):
    prev, depth, ringmask, bstage = state
    if bstage == B0:
        if char == "n" and char in bigram["["]:
            return (prev, depth, ringmask, BN)
        if char == "H" and char in bigram["["]:
            return (prev, depth, ringmask, BH)
        return None
    if bstage == BN:
        return (prev, depth, ringmask, BH) if char == "H" and char in bigram["n"] else None
    if bstage == BH:
        return ("]", depth, ringmask, OUTSIDE) if char == "]" and char in bigram["H"] else None

    if char not in bigram.get(prev, set()):
        return None
    if char == "(":
        return None if depth >= CAP else (char, depth + 1, ringmask, OUTSIDE)
    if char == ")":
        return None if depth == 0 else (char, depth - 1, ringmask, OUTSIDE)
    if char in "123456":
        return (char, depth, ringmask ^ (1 << (int(char) - 1)), OUTSIDE)
    if char == "[":
        return (prev, depth, ringmask, B0)
    return (char, depth, ringmask, OUTSIDE)


def compile_dfa(vocab, bigram, pre_eos):
    chars = sorted((c for c in vocab if len(c) == 1), key=vocab.get)
    initial = ("BOS", 0, 0, OUTSIDE)
    queue = deque([initial])
    seen = {initial}
    compiled = []
    while queue:
        state = queue.popleft()
        prev, depth, ringmask, bstage = state
        transitions = {}
        candidates = chars if bstage != OUTSIDE else sorted(bigram.get(prev, ()), key=vocab.get)
        for char in candidates:
            target = next_state(state, char, bigram)
            if target is None:
                continue
            transitions[str(vocab[char])] = state_name(target)
            if target not in seen:
                seen.add(target)
                queue.append(target)
        if bstage == OUTSIDE and depth == 0 and ringmask == 0 and prev in pre_eos:
            transitions[str(EOS_ID)] = "accept"
        compiled.append({"name": state_name(state), "transitions": transitions})

    states = [
        {"name": "start", "transitions": {str(BOS_ID): state_name(initial)}},
        *compiled,
        {"name": "accept", "transitions": {}, "accept": True},
    ]
    transition_count = sum(len(s["transitions"]) for s in states)
    assert len(states) <= 100_000, len(states)
    assert transition_count <= 1_000_000, transition_count
    return {
        "format": "mixlab.token_dfa",
        "version": 1,
        "vocab_size": VOCAB_SIZE,
        "start_state": "start",
        "eos_token_ids": [EOS_ID],
        "states": states,
    }, transition_count


def walk(dfa, smiles: str, vocab: dict[str, int], states=None):
    if states is None:
        states = {s["name"]: s for s in dfa["states"]}
    current = dfa["start_state"]
    bos_target = states[current]["transitions"].get(str(BOS_ID))
    if bos_target is None:
        return False, "missing-BOS-transition"
    current = bos_target
    for position, char in enumerate(smiles):
        token_id = vocab.get(char)
        if token_id is None:
            return False, f"unknown-char:{char}@{position}"
        target = states[current]["transitions"].get(str(token_id))
        if target is None:
            return False, f"missing-transition:{char}@{position}"
        current = target
    target = states[current]["transitions"].get(str(EOS_ID))
    if target is None or not states[target].get("accept", False):
        return False, "no-EOS-at-end"
    return True, "accepted"


def walk_accepts(dfa, smiles: str, vocab: dict[str, int]) -> bool:
    return walk(dfa, smiles, vocab)[0]


def replay(dfa, smiles_list, vocab):
    reasons = Counter()
    accepted = 0
    states = {s["name"]: s for s in dfa["states"]}
    for smiles in smiles_list:
        ok, reason = walk(dfa, smiles, vocab, states)
        accepted += ok
        if not ok:
            reasons[reason] += 1
    return accepted, reasons


def main():
    vocab = load_vocab()
    train = moses.get_dataset("train")
    split = int(len(train) * BUILD_FRACTION)
    build, holdout = train[:split], train[split:]
    bigram, pre_eos, build_max_depth, build_max_open_rings = corpus_stats(build)
    assert CAP >= build_max_depth, (CAP, build_max_depth)
    _, _, holdout_max_depth, holdout_max_open_rings = corpus_stats(holdout)

    dfa, num_transitions = compile_dfa(vocab, bigram, pre_eos)
    DFA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DFA_PATH.open("w") as f:
        json.dump(dfa, f, indent=2)
        f.write("\n")

    sanity_n = min(5000, len(build))
    sanity_accepted, sanity_reasons = replay(dfa, build[:sanity_n], vocab)
    holdout_accepted, rejection_reasons = replay(dfa, holdout, vocab)
    holdout_acceptance = holdout_accepted / len(holdout)

    positives = ["CC(C)Cc1ccccc1", "c1cc[nH]c1", "O=C(O)C", "CC(=O)Nc1ccccc1"]
    negatives = ["CC(", "c1ccccc", "CC=", "C()", "c1cc1cc1"]
    self_test_failures = []
    for smiles in positives:
        ok, reason = walk(dfa, smiles, vocab)
        print(f"SELFTEST positive {smiles}: {'PASS' if ok else 'FAIL'} ({reason})")
        if not ok:
            self_test_failures.append(("positive", smiles, reason))
    for smiles in negatives:
        ok, reason = walk(dfa, smiles, vocab)
        passed = not ok
        print(f"SELFTEST negative {smiles}: {'PASS' if passed else 'FAIL'} ({reason})")
        if not passed:
            self_test_failures.append(("negative", smiles, reason))

    report = {
        "holdout_acceptance": holdout_acceptance,
        "holdout_n": len(holdout),
        "build_max_depth": build_max_depth,
        "holdout_max_depth": holdout_max_depth,
        "num_states": len(dfa["states"]),
        "num_transitions": num_transitions,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
    }
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"build split: n={len(build)} max_depth={build_max_depth} max_open_rings={build_max_open_rings}")
    print(f"build sanity: {sanity_accepted}/{sanity_n} ({sanity_accepted / sanity_n:.6%}) reasons={dict(sanity_reasons)}")
    print(f"holdout: {holdout_accepted}/{len(holdout)} ({holdout_acceptance:.6%}) max_depth={holdout_max_depth} max_open_rings={holdout_max_open_rings}")
    print(f"rejection reasons: {dict(sorted(rejection_reasons.items()))}")
    print(f"DFA: states={len(dfa['states'])} transitions={num_transitions}")
    print(f"wrote {DFA_PATH.relative_to(ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")

    assert not self_test_failures, self_test_failures
    assert sanity_accepted == sanity_n, (sanity_accepted, sanity_n, sanity_reasons)
    assert holdout_acceptance >= 0.999, holdout_acceptance


if __name__ == "__main__":
    main()
