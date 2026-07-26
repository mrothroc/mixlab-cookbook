#!/usr/bin/env python3
"""The attention-baseline classifier for human_enhancers_cohn.

Loads a mixlab DNA LM exported to Hugging Face, then runs three arms x N seeds
and reports test accuracy, MCC, and AUROC:
  - pretrained-e2e : load exported HF weights, fine-tune everything (the baseline)
  - scratch        : same architecture, random init, fine-tune everything (control)
  - frozen         : load exported weights, freeze backbone, train head only

Used for the attention model, which exports to HF cleanly. The Mamba path uses
mixlab's native classification training instead (see ENHANCE_MAMBA3.md).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from transformers import AutoConfig, AutoModelForSequenceClassification

# nucleotide vocab, matching mixlab's fixed `dna` alphabet (see nucleotide_vocab.json
# written by `mixlab -mode prepare -input-format fasta`).
PAD, BOS, EOS = 0, 1, 2
BASE = {"A": 4, "C": 5, "G": 6, "T": 7, "N": 8}
SEQ_LEN = 512


def encode(seq):
    ids = [BOS] + [BASE.get(c, BASE["N"]) for c in seq.upper()] + [EOS]
    ids = ids[:SEQ_LEN]
    mask = [1] * len(ids)
    pad = SEQ_LEN - len(ids)
    ids += [PAD] * pad
    mask += [0] * pad
    return ids, mask


def load_split(split):
    from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn
    ds = HumanEnhancersCohn(split=split, version=0)
    X, M, Y = [], [], []
    for seq, lab in ds:
        ids, mask = encode(seq)
        X.append(ids); M.append(mask); Y.append(int(lab))
    return (torch.tensor(X, dtype=torch.long),
            torch.tensor(M, dtype=torch.long),
            torch.tensor(Y, dtype=torch.long))


def build_model(hf_dir, arm, device):
    if arm == "scratch":
        cfg = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(cfg, trust_remote_code=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_dir, trust_remote_code=True, num_labels=2)
    if arm == "frozen":
        for name, p in model.named_parameters():
            if "classifier" not in name and "score" not in name:
                p.requires_grad = False
    return model.to(device)


def run_arm(hf_dir, arm, seed, data, device, epochs, bs, lr):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Mtr, Ytr, Xte, Mte, Yte = data
    model = build_model(hf_dir, arm, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    loader = DataLoader(TensorDataset(Xtr, Mtr, Ytr), batch_size=bs, shuffle=True)
    model.train()
    for ep in range(epochs):
        t0 = time.time()
        tot = 0.0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(input_ids=xb, attention_mask=mb, labels=yb)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += out.loss.item()
        print(f"    [{arm} seed{seed}] epoch {ep+1}/{epochs} loss={tot/len(loader):.4f} ({time.time()-t0:.0f}s)", flush=True)
    # eval
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            xb = Xte[i:i+bs].to(device); mb = Mte[i:i+bs].to(device)
            logits = model(input_ids=xb, attention_mask=mb).logits
            preds.append(logits.argmax(-1).cpu())
            probs.append(torch.softmax(logits, -1)[:, 1].cpu())
    pred = torch.cat(preds).numpy()
    prob = torch.cat(probs).numpy()
    y = Yte.numpy()
    return {"arm": arm, "seed": seed,
            "test_acc": float(accuracy_score(y, pred)),
            "test_mcc": float(matthews_corrcoef(y, pred)),
            "test_auroc": float(roc_auc_score(y, prob))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--arms", nargs="+", default=["pretrained", "scratch", "frozen"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--frozen-lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/gate_summary.json")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)
    print("loading human_enhancers_cohn ...", flush=True)
    Xtr, Mtr, Ytr = load_split("train")
    Xte, Mte, Yte = load_split("test")
    print(f"train={len(Xtr)} test={len(Xte)} (balanced binary, 500bp)", flush=True)
    data = (Xtr, Mtr, Ytr, Xte, Mte, Yte)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            lr = args.frozen_lr if arm == "frozen" else args.lr
            print(f"=== arm={arm} seed={seed} lr={lr} ===", flush=True)
            rows.append(run_arm(args.hf_dir, arm, seed, data, device, args.epochs, args.bs, lr))
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(rows, open(args.out, "w"), indent=2)

    # summarize
    print("\n=== GATE SUMMARY ===", flush=True)
    summ = {}
    for arm in args.arms:
        accs = [r["test_acc"] for r in rows if r["arm"] == arm]
        mccs = [r["test_mcc"] for r in rows if r["arm"] == arm]
        aucs = [r["test_auroc"] for r in rows if r["arm"] == arm]
        summ[arm] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                     "mcc_mean": float(np.mean(mccs)), "auroc_mean": float(np.mean(aucs)), "n": len(accs)}
        print(f"{arm:12s} acc={np.mean(accs):.4f}±{np.std(accs):.4f}  mcc={np.mean(mccs):.4f}  auroc={np.mean(aucs):.4f}", flush=True)
    if "pretrained" in summ and "scratch" in summ:
        lift = summ["pretrained"]["acc_mean"] - summ["scratch"]["acc_mean"]
        spread = max(summ["pretrained"]["acc_std"], summ["scratch"]["acc_std"])
        passed = lift > spread
        print(f"\nlift(pretrained-scratch)={lift:+.4f}  max_seed_spread={spread:.4f}  GATE={'PASS' if passed else 'FAIL'}", flush=True)
        summ["gate"] = {"lift": lift, "spread": spread, "pass": bool(passed)}
    json.dump({"rows": rows, "summary": summ}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
