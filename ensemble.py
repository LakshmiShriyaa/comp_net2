"""Average the forecasts of several training runs (different --seed) into one results folder for routing.py.
    python ensemble.py --runs res_S4L res_S4L_s1 res_S4L_s2 --out res_ENS
All runs must use the same data, horizon and --times so their test windows match."""
import argparse, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
true = np.load(f"{a.runs[0]}/true_test.npy")
preds = []
for r in a.runs:
    if not np.array_equal(np.load(f"{r}/true_test.npy"), true):
        raise SystemExit(f"{r} has different test targets: runs are not comparable")
    preds.append(np.load(f"{r}/pred_test.npy"))
ens = np.mean(preds, axis=0)
mask = ~np.eye(true.shape[-1], dtype=bool)
rm = lambda p: float(np.sqrt(((p - true)[..., mask] ** 2).mean()))
for r, p in zip(a.runs, preds):
    print(f"{r:<20} RMSE (farthest step, off-diag) {rm(p):10.1f}")
print(f"{'ENSEMBLE':<20} RMSE {rm(ens):10.1f}")
np.save(f"{a.out}/pred_test.npy", ens)
shutil.copy(f"{a.runs[0]}/true_test.npy", f"{a.out}/true_test.npy")
