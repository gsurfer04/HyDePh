with open('/home/jamesc/HyDePhTest/SiC-4H-kk/test_seed_calc/pw.out') as f:
    text = f.read()

import re

for spin in ['SPIN UP', 'SPIN DOWN']:
    idx = text.rfind(f'------ {spin} ----------')
    part = text[idx:idx+3000]
    b_idx = part.find('bands (ev):')
    sub = part[b_idx+11:]
    # match all floats
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', sub)
    vals = [float(x) for x in nums]
    print(f'=== {spin} ===')
    for i in range(270, min(290, len(vals))):
        print(f'Band {i+1}: {vals[i]:.4f} eV')
