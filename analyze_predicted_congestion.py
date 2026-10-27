import csv
import numpy as np

forecast = np.loadtxt("ns3_forecast.csv", delimiter=",")
assert forecast.shape == (23, 23), f"Unexpected forecast shape: {forecast.shape}"

rows = []
with open("geant_link_capacities.csv", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        u = int(r["u_idx"])
        v = int(r["v_idx"])
        cap = float(r["capacity_kbps"])
        demand = (max(0, forecast[u, v]) + max(0, forecast[v, u])) / 2
        util = demand / cap if cap > 0 else 0
        rows.append((u, v, demand, cap, util))

rows.sort(key=lambda x: x[4], reverse=True)
print("FORECAST-BASED LINK RISK REPORT")
print("Link | Forecast proxy (kbps) | Capacity (kbps) | Proxy utilization")
for u, v, demand, cap, util in rows:
    print(f"{u}-{v} | {demand:.2f} | {cap:.2f} | {util:.2%}")

risk = [r for r in rows if r[4] >= 0.80]
print(f"\nLinks checked: {len(rows)}")
print(f"Links with proxy utilization >= 80%: {len(risk)}")
with open("predicted_congestion.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["u_idx", "v_idx", "forecast_proxy_kbps",
                "capacity_kbps", "proxy_utilization", "risk_80_percent"])
    for u, v, demand, cap, util in rows:
        w.writerow([u, v, round(demand, 3), round(cap, 3),
                    round(util, 6), util >= 0.80])
print("Saved: predicted_congestion.csv")
