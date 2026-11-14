#!/usr/bin/env python3
"""Confidence head on top of the final fusion model (frozen hp_A CNN + 28 descriptors, factorized decoding w=0.5).
Honest right/wrong labels via 5-fold cross-fitting per AR group inside the training part (outer fold). Head input: 128-d fusion
representation + output signals (MSP/margin/entropy of the joint head, MSP of Z/p/c, margin of the factorized score) + prototype
signals (cos to the predicted-class prototype, max cos, cos gap). 7-fold OOF evaluation: AUROC for wrong predictions, ECE,
coverage-accuracy. Its feature definitions (everything above the fusion model class) are reused by xgb_v7.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from hmi_umbra import paths

R = Path(paths.get("cls_root"))
EMB = R / "opt_v5/hp_A"
MORPH = R / "opt_v5/morph/morph_features.csv"
OUT = R / "opt_v5/conf"
dev = torch.device("cuda")
EPOCHS, PATIENCE, BS, LR, WD, LS, MOD_DROP, INNER = 300, 30, 256, 1e-3, 1e-3, 0.12, 0.3, 5
Z_CL, P_CL, C_CL = list("ABCDEFH"), list("xrsahk"), list("xoic")
LOG_COLS = ["grp_umb_deg2", "grp_pen_deg2", "ls_umb_deg2", "ls_pen_deg2", "ls_pen_umb_ratio",
            "grp_n_regions", "grp_n_reg_umb", "grp_n_reg_pen", "ls_n_umb", "pen_end_min", "pen_end_max"]
LIN_COLS = ["grp_major_deg", "grp_minor_deg", "grp_frac_largest", "grp_bins_filled", "ls_major_deg", "ls_minor_deg",
            "ls_axis_ratio", "ls_ecc", "ls_ray_cov", "ls_ray_gap_deg_ang", "ls_pen_width_med_deg", "ls_pen_width_p10_deg",
            "ls_trans_mean", "ls_edge_ph_frac", "ls_umb_offset_norm", "largest_pos_dev"]
M = pd.read_csv(MORPH)
M["pen_end_min"] = M[["grp_pen_endA_deg2", "grp_pen_endB_deg2"]].min(axis=1)
M["pen_end_max"] = M[["grp_pen_endA_deg2", "grp_pen_endB_deg2"]].max(axis=1)
M["largest_pos_dev"] = (M["grp_largest_pos"] - 0.5).abs()
X = pd.DataFrame({"stem": M["stem"]})
for c in LOG_COLS:
    X[c] = np.log1p(M[c].clip(lower=0))
for c in LIN_COLS:
    X[c] = M[c]
FEAT = LOG_COLS + LIN_COLS
for c in FEAT:
    med, iqr = X[c].median(), (X[c].quantile(0.75) - X[c].quantile(0.25)) or 1.0
    X[c] = ((X[c] - med) / iqr).clip(-5, 5).fillna(0.0)
X["has_umb"] = M["ls_ray_cov"].notna().astype(float)
FEAT += ["has_umb"]
FMAP = dict(zip(X["stem"], X[FEAT].to_numpy(np.float32)))
KF = len(FEAT)


class Fusion(nn.Module):
    def __init__(self, n_cls, d_emb=528):
        super().__init__()
        self.cnn = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_emb, 64), nn.ReLU())
        self.fea = nn.Sequential(nn.Linear(KF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3))
        self.hj, self.hz, self.hp, self.hc = nn.Linear(128, n_cls), nn.Linear(128, 7), nn.Linear(128, 6), nn.Linear(128, 4)
        self.az, self.apz, self.ac = nn.Linear(64, 7), nn.Linear(64, 6), nn.Linear(64, 4)

    def forward(self, e, f, drop_cnn=None):
        hc = self.cnn(e)
        if drop_cnn is not None:
            hc = hc * (~drop_cnn).float().unsqueeze(1)
        hf = self.fea(f)
        h = self.mix(torch.cat([hc, hf], 1))
        return self.hj(h), self.hz(h), self.hp(h), self.hc(h), self.az(hf), self.apz(hf), self.ac(hf), h


def load(k, part):
    d = np.load(EMB / f"fold{k}" / f"{part}_outputs.npz", allow_pickle=True)
    labels = list(d["labels"]); lab = np.array(labels)[d["y"]]
    t = torch.tensor
    return {"stem": d["stem"], "group": d["group_id"], "e": t(d["emb"]).float(), "f": t(np.stack([FMAP[s] for s in d["stem"]])),
            "y": t(d["y"]).long(), "z": t([Z_CL.index(l[0]) for l in lab]), "p": t([P_CL.index(l[1]) for l in lab]),
            "c": t([C_CL.index(l[2]) for l in lab]), "labels": labels}


def sub(D, idx):
    return {k: (v[idx] if isinstance(v, (torch.Tensor, np.ndarray)) else v) for k, v in D.items()}


def train_fusion(Tr, Va, n_cls, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cnt = np.bincount(Tr["y"].numpy(), minlength=n_cls).astype(float)
    w = 1 / np.sqrt(np.maximum(cnt, 1)); w = torch.tensor(np.minimum(w / w.mean(), 5.0), dtype=torch.float32)
    m = Fusion(n_cls).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    ce_j, ce = nn.CrossEntropyLoss(weight=w.to(dev), label_smoothing=LS), nn.CrossEntropyLoss(label_smoothing=LS)
    best, bad, state, n = -1, 0, None, len(Tr["y"])
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n)
        for s in range(0, n, BS):
            i = perm[s:s + BS]
            o = m(Tr["e"][i].to(dev), Tr["f"][i].to(dev), torch.rand(len(i), device=dev) < MOD_DROP)
            yy = [Tr[kk][i].to(dev) for kk in ("y", "z", "p", "c")]
            loss = 0.5 * ce_j(o[0], yy[0]) + 0.2 * sum(ce(o[j], yy[j]) for j in (1, 2, 3)) + 0.1 * sum(ce(o[j + 3], yy[j]) for j in (1, 2, 3))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = m(Va["e"].to(dev), Va["f"].to(dev))[0].argmax(1).cpu().numpy()
        f1 = f1_score(Va["y"].numpy(), pv, average="macro", labels=list(range(n_cls)), zero_division=0)
        if f1 > best:
            best, bad, state = f1, 0, {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state); m.eval()
    return m


def infer(m, D):
    with torch.no_grad():
        o = m(D["e"].to(dev), D["f"].to(dev))
    return [F.softmax(x, 1).cpu().numpy() for x in o[:4]], o[7].cpu().numpy()


lg = lambda p: np.log(np.clip(p, 1e-12, None))
l2n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-9)


def signals(P, h, proto, labels):
    Zi = np.array([Z_CL.index(l[0]) for l in labels]); Pi = np.array([P_CL.index(l[1]) for l in labels]); Ci = np.array([C_CL.index(l[2]) for l in labels])
    s = 0.5 * lg(P[0]) + 0.5 * (lg(P[1])[:, Zi] + lg(P[2])[:, Pi] + lg(P[3])[:, Ci])
    pred = s.argmax(1)
    ss = np.sort(s, 1)[:, ::-1]
    pj = np.sort(P[0], 1)[:, ::-1]
    cos = l2n(h) @ proto.T
    cs = np.sort(cos, 1)[:, ::-1]
    sig = np.column_stack([pj[:, 0], pj[:, 0] - pj[:, 1], (P[0] * lg(P[0])).sum(1),
                           P[1].max(1), P[2].max(1), P[3].max(1), ss[:, 0] - ss[:, 1],
                           cos[np.arange(len(pred)), pred], cs[:, 0], cos[np.arange(len(pred)), pred] - np.where(cos.argmax(1) == pred, cs[:, 1], cs[:, 0])])
    return pred, s, sig


def protos(m, D, n_cls):
    _, h = infer(m, D)
    h = l2n(h)
    return np.stack([l2n(h[D["y"].numpy() == c].mean(0, keepdims=True))[0] if (D["y"].numpy() == c).any() else np.zeros(h.shape[1]) for c in range(n_cls)])


class Conf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_conf(Xc, yc, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-6
    Xn = torch.tensor((Xc - mu) / sd, dtype=torch.float32); yt = torch.tensor(yc, dtype=torch.float32)
    n = len(yt); perm = torch.randperm(n); nv = n // 10
    va, tr = perm[:nv], perm[nv:]
    m = Conf(Xn.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    best, bad, state = 1e9, 0, None
    for ep in range(500):
        m.train()
        p2 = tr[torch.randperm(len(tr))]
        for s in range(0, len(p2), 256):
            i = p2[s:s + 256]
            loss = F.binary_cross_entropy_with_logits(m(Xn[i].to(dev)), yt[i].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(m(Xn[va].to(dev)), yt[va].to(dev)).item()
        if vl < best:
            best, bad, state = vl, 0, {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    m.load_state_dict(state); m.eval()
    return lambda Xq: torch.sigmoid(m(torch.tensor((Xq - mu) / sd, dtype=torch.float32).to(dev))).detach().cpu().numpy()


rows = []
for k in range(7):
    Tr, Va, Te = load(k, "train"), load(k, "val"), load(k, "test")
    labels = Tr["labels"]; n_cls = len(labels)
    yT = Tr["y"].numpy()
    oof_h, oof_sig, oof_ok = np.zeros((len(yT), 128)), None, np.zeros(len(yT))
    for j, (itr, iho) in enumerate(GroupKFold(n_splits=INNER).split(np.zeros(len(yT)), yT, Tr["group"])):
        mi = train_fusion(sub(Tr, itr), Va, n_cls, 100 * k + j)
        pr = protos(mi, sub(Tr, itr), n_cls)
        P, h = infer(mi, sub(Tr, iho))
        pred, _, sig = signals(P, h, pr, labels)
        oof_h[iho] = h
        oof_sig = np.zeros((len(yT), sig.shape[1])) if oof_sig is None else oof_sig
        oof_sig[iho] = sig
        oof_ok[iho] = (pred == yT[iho]).astype(float)
    mf = train_fusion(Tr, Va, n_cls, 100 * k + 99)
    pr = protos(mf, Tr, n_cls)
    P, h = infer(mf, Te)
    pred, s, sig = signals(P, h, pr, labels)
    conf = train_conf(np.hstack([oof_h, oof_sig]), oof_ok)(np.hstack([h, sig]))
    yt = Te["y"].numpy()
    rows.append(pd.DataFrame({"fold": k, "stem": Te["stem"], "true": np.array(labels)[yt], "pred": np.array(labels)[pred],
                              "ok": pred == yt, "conf_head": conf, "MSP": sig[:, 0], "margin": sig[:, 1], "neg_entropy": sig[:, 2],
                              "factorized_margin": sig[:, 6], "cos_proto_pred": sig[:, 7], "cos_gap": sig[:, 9],
                              "top5": [" ".join(np.array(labels)[np.argsort(-s[i])[:5]]) for i in range(len(yt))]}))
    np.savez(OUT / f"fold{k}_conf.npz", stem=Te["stem"], y=yt, labels=np.array(labels), score=s, h=h, proto=pr,
             pj=P[0], pz=P[1], pp=P[2], pc=P[3], conf=conf)
    print(f"fold {k}: acc {np.mean(pred == yt):.4f} | inner-OOF acc train {oof_ok.mean():.4f} | AUROC conf_head "
          f"{roc_auc_score(1 - (pred == yt), -conf):.4f}", flush=True)

D = pd.concat(rows, ignore_index=True)
D.to_csv(OUT / "conf_oof.csv", index=False)
wrong = (~D.ok).astype(int).to_numpy()
print(f"\nOOF 7-fold n={len(D)} | accuracy (fusion 1 seed, factorized) {D.ok.mean():.4f}")
print("\nAUROC for detecting wrong predictions:")
for c in ["conf_head", "factorized_margin", "MSP", "margin", "neg_entropy", "cos_proto_pred", "cos_gap"]:
    print(f"  {c:<18} {roc_auc_score(wrong, -D[c].to_numpy()):.4f}")


def ece(conf, ok, nb=15):
    b = np.linspace(0, 1, nb + 1); e = 0.0
    for lo, hi in zip(b[:-1], b[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


ok = D.ok.to_numpy().astype(float)
print(f"\nECE: conf_head {ece(D.conf_head.to_numpy(), ok):.4f} (mean {D.conf_head.mean():.4f}) | MSP {ece(D.MSP.to_numpy(), ok):.4f} "
      f"(mean {D.MSP.mean():.4f}) | accuracy {ok.mean():.4f}")
cov = []
for hold in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    r = {"held_%": int(hold * 100)}
    for c in ["conf_head", "factorized_margin", "MSP", "cos_proto_pred"]:
        v = D[c].to_numpy()
        acc = v > np.quantile(v, hold) if hold > 0 else np.ones(len(v), bool)
        r[f"acc_{c}"] = round(float(ok[acc].mean()), 4)
    cov.append(r)
pd.set_option("display.width", 220)
print("\nCoverage vs accuracy (accuracy of ACCEPTED crops):")
print(pd.DataFrame(cov).to_string(index=False))
print(f"saved: {OUT}")
