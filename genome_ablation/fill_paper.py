"""Write the tab:genome rows from make_table.py into paper_access/main.tex,
between %GENOME_ROWS_BEGIN / %GENOME_ROWS_END (first run: replaces %GENOME_ROWS%)."""
import subprocess, sys, re
P = r'C:\Neural-Networks-Files\torchquantum\paper_access\main.tex'
out = subprocess.run([sys.executable, __file__.replace('fill_paper.py', 'make_table.py')],
                     capture_output=True, text=True).stdout
rows = [l for l in out.splitlines() if l.rstrip().endswith('\\')]
names = {'p0.05': r'$p_m=0.05$', 'p0.1': r'$p_m=0.1$', 'p0.2': r'$p_m=0.2$', 'p0.4': r'$p_m=0.4$'}
fixed = []
for r in rows:
    head = r.strip().split(' & ')[0]
    fixed.append('        ' + names.get(head, head) + r.strip()[len(head):])
block = '%GENOME_ROWS_BEGIN\n' + '\n'.join(fixed) + '\n%GENOME_ROWS_END'
s = open(P, encoding='utf-8').read()
if '%GENOME_ROWS%' in s:
    s = s.replace('%GENOME_ROWS%', block)
else:
    s = re.sub(r'%GENOME_ROWS_BEGIN.*?%GENOME_ROWS_END', lambda m: block, s, flags=re.S)
open(P, 'w', encoding='utf-8').write(s)
print(block)
