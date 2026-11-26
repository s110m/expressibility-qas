"""Do either search proxy rank architectures better than circuit size alone?

The four-arm result is confounded: loss_succ and expressibility select nearly
disjoint circuit sizes, so their accuracy difference is mostly a capacity
difference. This asks the question directly, on a common set of 32 architectures
spanning 6-192 params where both proxies are defined and from-scratch accuracy
is known.

Ground truth is noiseless validation accuracy after 200 epochs from scratch --
the same axis QuantumNAS's Figure 9 uses, and the one CLAUDE.md's +0.17/+0.67
Spearman figures are quoted against. It is NOT noisy test accuracy; a proxy
could in principle rank noise robustness well and this axis poorly.

Both proxies are minimised by their searches, so a NEGATIVE rho means the proxy
is aligned with the outcome.

    python compare_proxies_32.py --in expr32.json
"""
import argparse
import json
import math


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float('nan')


def spearman(x, y):
    return pearson(rank(x), rank(y))


def partial(x, y, z):
    """Spearman of x,y with z partialled out (rank-level partial correlation)."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den else float('nan')


def fisher_ci(r, n, k=0, conf=1.96):
    """CI on a (partial) correlation; k = number of variables partialled out."""
    if abs(r) >= 1 or n - k - 3 <= 0:
        return float('nan'), float('nan')
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - k - 3)
    return math.tanh(z - conf * se), math.tanh(z + conf * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='expr32.json')
    args = ap.parse_args()
    rows = json.load(open(args.inp))
    n = len(rows)

    acc = [r['acc_valid'] for r in rows]
    lp = [math.log(r['params']) for r in rows]
    proxies = {'loss_succ (inherited loss)': [r['inh_loss'] for r in rows],
               'expressibility (KL)': [r['expr_kl'] for r in rows]}

    print(f"n = {n} architectures, params {min(r['params'] for r in rows)}"
          f"-{max(r['params'] for r in rows)}\n")

    r = spearman(lp, acc)
    lo, hi = fisher_ci(r, n)
    print(f"size baseline    rho(ln params, from-scratch acc) = {r:+.3f}  "
          f"[{lo:+.2f},{hi:+.2f}]")
    print("                 (this is what any proxy has to beat)\n")

    print("proxy vs outcome  -- NEGATIVE rho = proxy aligned with accuracy")
    for name, p in proxies.items():
        r = spearman(p, acc)
        lo, hi = fisher_ci(r, n)
        print(f"  {name:28} rho = {r:+.3f}  [{lo:+.2f},{hi:+.2f}]")

    print("\nproxy vs size     -- how much of the proxy is just circuit size?")
    for name, p in proxies.items():
        print(f"  {name:28} rho = {spearman(p, lp):+.3f}")

    print("\npartial, size held fixed  -- the decisive number")
    for name, p in proxies.items():
        r = partial(p, acc, lp)
        lo, hi = fisher_ci(r, n, k=1)
        sig = "" if lo < 0 < hi else "  *"
        print(f"  {name:28} rho = {r:+.3f}  [{lo:+.2f},{hi:+.2f}]{sig}")
    print("\n  * = 95% CI excludes zero")


if __name__ == '__main__':
    main()
