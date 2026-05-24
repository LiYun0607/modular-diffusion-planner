"""Split large_pref_pairs.jsonl into train/val by sampler-balanced 80/20."""
import json, random, os
pairs = [json.loads(l) for l in open('/root/corl_work/outputs/large_pref_pairs.jsonl') if l.strip()]
print(f'total pairs: {len(pairs)}')
random.seed(2026); random.shuffle(pairs)
n_train = int(len(pairs) * 0.8)
train = pairs[:n_train]; val = pairs[n_train:]
print(f'split: train={len(train)}, val={len(val)}')
# also report by sampler
from collections import Counter
print(f'sampler counts: total={dict(Counter(p["sampler"] for p in pairs))}  train={dict(Counter(p["sampler"] for p in train))}  val={dict(Counter(p["sampler"] for p in val))}')
with open('/root/corl_work/outputs/large_pref_pairs_train.jsonl','w') as f:
    for p in train: f.write(json.dumps(p) + '\n')
with open('/root/corl_work/outputs/large_pref_pairs_val.jsonl','w') as f:
    for p in val: f.write(json.dumps(p) + '\n')
print('done')
