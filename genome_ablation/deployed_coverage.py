"""Distinct layouts each search evaluated for the architecture it deployed.

Read from the search log (every fresh evaluation is logged), excluding the
post-search enumeration block at the end of a finished log."""
import ast, glob, json, os, re, collections
import numpy as np
GA = os.path.dirname(os.path.abspath(__file__))
pat = re.compile(r"Solution: (\{.*?\}), objectives: (\{.*?\})")
DEV = {'FakeQuitoV2': 'quito', 'FakeAthensV2': 'athens', 'FakeBelemV2': 'belem'}
def trim(a): return tuple(a[:a[-1] * 2]) + (a[-1],)
out = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(GA, '*', '*_s1*.json'))):
    if f.endswith(('partial.json', 'fromlog.json')):
        continue
    d = json.load(open(f))
    log = os.path.join(GA, 'logs', f"{d['setting']}_{DEV[d['backend']]}_{d['seed']}.log")
    recs = [ast.literal_eval(m.group(1)) for m in map(pat.search, open(log, encoding='utf-8', errors='replace')) if m]
    tail = 0
    while tail < len(recs) and trim(recs[-1 - tail]['arch']) == trim(recs[-1]['arch']):
        tail += 1
    if tail >= 60:
        recs = recs[:-tail]
    dep = trim(d['deployed']['solution']['arch'])
    lays = {tuple(r['layout']) for r in recs if trim(r['arch']) == dep}
    out[d['setting']].append((len(lays), d['deployed']['success_rate_rank']))
res = {}
for s, v in sorted(out.items()):
    a = np.array(v)
    res[s] = {'n': len(v), 'layouts_tried_mean': float(a[:, 0].mean()),
              'layouts_tried_median': float(np.median(a[:, 0])), 'per_run': v}
    print(f"{s:10s} n={len(v)} layouts tried for deployed arch: mean {a[:,0].mean():5.1f} median {np.median(a[:,0]):4.0f}  pairs {v}")
json.dump(res, open(os.path.join(GA, 'deployed_coverage.json'), 'w'), indent=1)
