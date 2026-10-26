"""Build the timeline for a dataset converted from XML files named ...YYYY-MM-DD-HH-MM...xml.

Writes one timestamp (minutes since 1970-01-01) per step, in the same sorted-filename order
convert_xml_datasets.py used, and reports every gap in the timeline.

Usage (run in your WAN folder)
    python make_times.py --pattern "data/traffic-matrices/*.xml" --out geant_times.npy --check geant_final.npy
"""
import argparse
import datetime as dt
import glob
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", default="data/traffic-matrices/*.xml")
ap.add_argument("--out", default="geant_times.npy")
ap.add_argument("--step", type=int, default=15, help="expected minutes between steps")
ap.add_argument("--check", default=None, help="optional .npy whose length must match the number of files")
args = ap.parse_args()

files = sorted(glob.glob(args.pattern))
if not files:
    raise SystemExit(f"No files match {args.pattern}")
pat = re.compile(r"(\d{4})-(\d\d)-(\d\d)-(\d\d)-(\d\d)")
stamps = []
for f in files:
    m = pat.search(os.path.basename(f))
    if m is None:
        raise SystemExit(f"No timestamp in file name: {f}")
    stamps.append(dt.datetime(*map(int, m.groups())))
minutes = np.array([int((s - dt.datetime(1970, 1, 1)).total_seconds() // 60) for s in stamps], dtype=np.int64)
np.save(args.out, minutes)

T = len(minutes)
print(f"{T} steps, {stamps[0]} -> {stamps[-1]}; saved {args.out}")
if args.check:
    n = np.load(args.check, mmap_mode="r").shape[0]
    print(f"length check against {args.check}: data has {n} steps -> {'OK' if n == T else 'MISMATCH - do not use'}")

diff = np.diff(minutes)
bad = np.where(diff != args.step)[0]
t_end, v_end = int(T * 0.7), int(T * 0.85)
where = lambda i: "train" if i < t_end else ("val" if i < v_end else "test")
missing = 0
print(f"\nirregular spacings: {len(bad)}")
for i in bad:
    miss = int(diff[i] // args.step - 1)
    missing += miss
    print(f"  between step {i} and {i + 1} ({where(i + 1)} split): {stamps[i]} -> {stamps[i + 1]}   "
          f"(~{miss} step(s) missing, {diff[i] / 60:.1f} h apart)")
print(f"total missing steps (approx): {missing}  = {missing * args.step / 1440:.1f} days")
print(f"splits used by train.py: train 0-{t_end - 1}, val {t_end}-{v_end - 1}, test {v_end}-{T - 1}")
