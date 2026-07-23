# Get molecular validity without retraining

If you want to tweak an existing SMILES model without retraining, another strategy is to improve it at sampling time. A structural token DFA (deterministic finite automaton that only allows the next tokens that keep the string well formed) keeps parentheses, ring digits, bracket atoms, bonds, and adjacency well formed. RDKit then filters the small residue that a context-free grammar cannot decide.

## Experimental Demonstration

I ran an experiment to test, using 12,000 constrained samples at T=1.0.

| Surface | Metric | Value | vs baseline |
|---|---|---|---|
| **(b) structural** (grammar support guarantee) | structural-valid rate on enforced subset | **1.0000** | - |
| **(b) full RDKit** (constrained only) | RDKit-valid rate | **0.9941** | 0.9875 unconstrained |
| **(d) post-hoc verifier** (constrained + RDKit filter) | delivered-valid rate | **1.0000** | - |
| coverage guard | Novelty | 0.7766 | band 0.747 to 0.847 ok |
| coverage guard | IntDiv | 0.8564 | band 0.837 to 0.877 ok |

All 12,000 outputs are structurally valid. RDKit rejects 71: kekulization 52, valence 16, and parse 3. Filtering those leaves 11,929 delivered molecules at 1.0000 validity. Full results are in `results/constrained_12k.metrics.json` and `results/constrained_12k.results.json`.

A grammar can guarantee balanced parentheses, matched ring digits, well-formed bracket atoms, and legal adjacency. Valence and aromatic-ring kekulizability need a chemical verifier. Rothrock 2026, section 8 describes this grammar and verifier split; `SELFIES_LEG.md` shows the representation alternative.

## How the DFA is built

`scripts/build_smiles_dfa.py` compiles a paren-depth counter capped at 8, a 6-bit ring-open bitmask for digits 1 to 6, a bracket sub-DFA for `[nH]` and `[H]`, and a bond-pending flag. It learns allowed atom and bond adjacency from MOSES character bigrams and start, pre-EOS, and after-bond sets. The result has 32,902 states and 170,199 transitions in `grammars/smiles_structural.token_dfa.json`.

The ring toggles matter because 709k MOSES molecules reuse a digit. EOS is allowed only at depth 0, with no open rings, outside brackets, and with no pending bond.

To check that the grammar does not exclude ordinary molecules, the build script replays the held-out MOSES set. Acceptance is **0.99998**, with 3 rejections in 158,467 caused by rare `[@…]` chirality forms outside the vocabulary. An independent checker measures structural validity on generated samples.

## Implementation note

This uses a hand-built DFA in Python (`scripts/build_smiles_dfa.py`) because it is an exploration that needed direct control over the bounded state machine. You would not hand-roll it in production. This is the classic lex-and-yacc problem (tokenize, then a grammar that balances parentheses and rings), and modern tools compile a grammar into the token mask for you:

- In mixlab, prefer the grammar route (`-grammar` / `-grammar-string`, a GBNF context-free grammar) over the precompiled `-grammar-table`. A grammar expresses the balanced structure directly, with a real stack, so there is no depth cap to pick.
- Outside mixlab, libraries such as Outlines, XGrammar, and llama.cpp GBNF take a regex, a context-free grammar, or a JSON schema and build the finite-state token mask automatically.

The precompiled DFA table is useful only when the structure is shallow and bounded, as SMILES nesting is here, and you want a fast per-step mask lookup. Otherwise, write the grammar once and let the tool compile it.

## Run it

Run from this recipe directory with mixlab >= v0.73.0 on PATH and a Python environment containing RDKit and MOSES. This leg constrains the SMILES model from [MOLGPT_REPRODUCTION.md](MOLGPT_REPRODUCTION.md), so run that leg first to produce `checkpoints/molgpt_record.safetensors`.

```bash
export MIXLAB_MLX_CACHE_LIMIT_MB=4096
# 1. compile + validate the DFA (holdout acceptance >= 0.999 gate is enforced in-script)
python scripts/build_smiles_dfa.py            # -> grammars/smiles_structural.token_dfa.json + dfa_build_report.json
# 2. constrained-sample + score the two surfaces
python scripts/constrained_sample.py --n 12000 --n-jobs 8 --out samples/constrained_12k.smi
#    -> structural-valid 1.000, constrained-only ~0.994, +filter 1.000, Novelty/IntDiv within band
```
