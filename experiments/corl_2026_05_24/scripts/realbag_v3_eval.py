"""Apply V3 proxy to real-vehicle ego trajectories extracted from 5-01 R1 bags."""
import sys, json, os, math
sys.path.insert(0, '/root/corl_work/scripts')
from cc_proxy_v3 import score_trajectory_v3, CCProxyConfig, aggregate_bucket_results
import numpy as np

cfg = CCProxyConfig()
bags = [
    ('R1_rep1_baseline',         '/root/corl_work/outputs/real_trajs_v2/R1_rep1_baseline.jsonl'),
    ('R1_rep1_kashiwa_selora',   '/root/corl_work/outputs/real_trajs_v2/R1_rep1_kashiwa_selora.jsonl'),
    ('R1_rep2_kashiwa_selora',   '/root/corl_work/outputs/real_trajs_v2/R1_rep2_kashiwa_selora.jsonl'),
]

out_all = {}
for tag, path in bags:
    if not os.path.exists(path): continue
    records = [json.loads(l) for l in open(path) if l.strip()]
    per_traj = []
    for r in records:
        fut = r['ego_future_egoframe']  # [(x, y, h, v), ...]
        if len(fut) < 10: continue
        ego_xy = [(p[0], p[1], p[2], i*0.1) for i, p in enumerate(fut)]
        ego_v_provided = [p[3] for p in fut]
        v0 = r['ego_v_now']
        s = score_trajectory_v3(ego_xy=ego_xy, ego_v_provided=ego_v_provided, ego_v0=v0, dt=0.1, cfg=cfg)
        per_traj.append(s)
    agg = aggregate_bucket_results(per_traj)
    out_all[tag] = agg
    ov = agg['_overall']
    print(f'\n=== {tag} (n={ov["n"]}):')
    print(f'  strict={ov["strict_safety_violation_rate"]*100:5.1f}% drv={ov["driverlike_violation_rate"]*100:5.1f}% cmf={ov["comfort_violation_rate"]*100:5.1f}%')
    print(f'  per_criterion (>1%):')
    for k, v in sorted(ov['per_criterion_rate'].items(), key=lambda x: -x[1]):
        if v > 0.01: print(f'    {k:<35s} {v*100:5.1f}%')
    print(f'  per-bucket:')
    for b, st in agg.items():
        if b == '_overall': continue
        print(f'    {b:<22s} n={st["n"]:4d}  strict={st["strict_safety_violation_rate"]*100:5.1f}%  drv={st["driverlike_violation_rate"]*100:5.1f}%  cmf={st["comfort_violation_rate"]*100:5.1f}%')

json.dump(out_all, open('/root/corl_work/outputs/realbag_v3_eval.json','w'), indent=2, default=float)
print(f'\nSaved /root/corl_work/outputs/realbag_v3_eval.json')
