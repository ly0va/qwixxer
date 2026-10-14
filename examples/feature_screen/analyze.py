"""Offline re-analysis of the feature_screen CSV (spike). Usage: python3 analyze.py feature_screen.csv [GM=gate env]"""
import csv, sys
import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
F = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

# Recover decision blocks: candidates are contiguous, sorted by z desc, one pick each, <= 4.
blocks, cur = [], []
for i, r in enumerate(rows):
    new = False
    if cur:
        p = rows[cur[-1]]
        has_pick = any(rows[j]["is_pick"] == "1" for j in cur)
        if (r["game"], r["phase"], r["cdiff"]) != (p["game"], p["phase"], p["cdiff"]):
            new = True
        elif float(r["z"]) > float(p["z"]) or len(cur) == 4 or (r["is_pick"] == "1" and has_pick):
            new = True
    if new:
        blocks.append(cur)
        cur = []
    cur.append(i)
blocks.append(cur)
blocks = [b for b in blocks if sum(rows[j]["is_pick"] == "1" for j in b) == 1]
print(f"{len(blocks)} decision blocks recovered")

n_games = int(F["game"].max()) + 1
mass = np.stack([F[f"mass{r}"] for r in range(4)], 1)
mass_tot = mass.sum(1)

def feats(name, c, p):
    d = lambda k: F[k][c] - F[k][p]
    cd = F["cdiff"][p]
    if name == "ceiling":
        return [d("w_b") - d("p")]
    if name == "length":
        return [d("turns"), d("end2"), d("end4")]
    if name == "length_x_cdiff":
        s = np.tanh(cd / 10)
        return [d("turns"), d("end2"), d("end4"), s * d("turns"), s * d("end2"), s * d("end4")]
    if name == "mass_rows":
        return [mass[c, r] - mass[p, r] for r in range(4)]
    if name == "mass_total":
        return [mass_tot[c] - mass_tot[p]]
    if name == "mass_total_x_turns":
        # value of remaining space should scale with remaining game
        return [mass_tot[c] - mass_tot[p], (mass_tot[c] - mass_tot[p]) * F["turns"][p] / 20]
    if name == "phase24":
        return [d("fs_us"), d("fs_opp"), d("thr_us"), d("thr_opp")]
    if name == "lockcal":
        return [d("lockcal")]
    raise KeyError(name)

GROUPS = ["ceiling", "length", "length_x_cdiff", "mass_rows", "mass_total", "mass_total_x_turns", "phase24", "lockcal"]

def run(train_pred, label):
    # pair rows
    pairs = []  # (block idx, c, p)
    for bi, b in enumerate(blocks):
        p = next(j for j in b if rows[j]["is_pick"] == "1")
        for c in b:
            if c != p:
                pairs.append((bi, c, p))
    bi_arr = np.array([x[0] for x in pairs])
    c_arr = np.array([x[1] for x in pairs]); p_arr = np.array([x[2] for x in pairs])
    games = F["game"][p_arr].astype(int)
    tr = train_pred(games); te = ~tr
    y = (F["w_a"][c_arr] - F["w_a"][p_arr]) - (F["p"][c_arr] - F["p"][p_arr])
    var_y = y[te].var()
    print(f"\n== {label}: {len(pairs)} pairs, test {te.sum()}, var(y) {var_y:.5f}")
    print(f"{'group':<22}{'R2 test':>9}{'gain/pg':>10}{'SE':>8}   coefs")
    test_games = sorted(set(games[te]))
    for g in GROUPS:
        X = np.array(feats(g, c_arr, p_arr)).T
        A = np.c_[np.ones(len(X)), X]
        b, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        r2 = 1 - ((y[te] - A[te] @ b) ** 2).mean() / var_y
        # re-pick with P + beta . dF (no intercept) per test decision
        adj = X @ b[1:]
        per_game = np.zeros(n_games)
        best = {}
        for k in np.where(te)[0]:
            bi = bi_arr[k]
            sc = F["p"][c_arr[k]] + adj[k]
            if sc > best.get(bi, (F["p"][p_arr[k]], None))[0]:
                best[bi] = (sc, k)
        for bi, (sc, k) in best.items():
            if k is not None:
                per_game[games[k]] += F["w_a"][c_arr[k]] - F["w_a"][p_arr[k]]
        v = per_game[test_games] / 2  # per player-game
        print(f"{g:<22}{r2:>9.4f}{v.mean():>+10.4f}{v.std(ddof=1)/np.sqrt(len(v)):>8.4f}   {np.round(b[1:], 4)}")

run(lambda g: g < n_games // 2, "train first half, test second")
run(lambda g: g >= n_games // 2, "train second half, test first")
run(lambda g: g % 2 == 0, "train even games, test odd")

# ---- search-side: gated raw argmax vs shrinkage toward the net ----
def policy_gain(score_fn, sel_blocks=None):
    per_game = np.zeros(n_games)
    for bi, b in enumerate(blocks):
        p = next(j for j in b if rows[j]["is_pick"] == "1")
        j = max(b, key=lambda c: score_fn(b, c, p))
        per_game[int(F["game"][p])] += F["w_a"][j] - F["w_a"][p]
    v = per_game / 2
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

import os
GM = float(os.environ.get("GM", "0.15"))
def gap(b):
    zs = sorted((F["z"][c] for c in b), reverse=True)
    return zs[0] - zs[1]
print("\n== search-side picks on ALL games (W_B picks, scored on W_A)")
for name, fn in [
    ("raw argmax W_B", lambda b, c, p: F["w_b"][c]),
    (f"gated (z gap < {GM}) argmax W_B", lambda b, c, p: F["w_b"][c] if gap(b) < GM else (1.0 if c == p else 0.0)),
] + [(f"shrink lam={l}", (lambda l: lambda b, c, p: F["p"][c] + l * (F["w_b"][c] - F["p"][c]))(l)) for l in (0.3, 0.5, 0.63, 0.8)]:
    m, se = policy_gain(fn)
    print(f"  {name:<34} {m:+.4f} ± {se:.4f}")
