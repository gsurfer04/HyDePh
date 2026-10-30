import numpy as np
import ase.io
from hydeph.defect_builder import mic_displacement

es = ase.io.read("relaxed_es_delta_scf.xyz")
cell = es.cell

c3 = [13, 29, 77]
si3 = [8, 56, 72]

print("In relaxed_es_delta_scf.xyz:")
print("C atoms:")
for i in range(3):
    for j in range(i+1, 3):
        dr = mic_displacement(es.positions[c3[i]:c3[i]+1], es.positions[c3[j]:c3[j]+1], cell)
        d = np.linalg.norm(dr)
        print(f"  d(C{c3[i]} - C{c3[j]}) = {d:.4f} A")

print("Si atoms:")
for i in range(3):
    for j in range(i+1, 3):
        dr = mic_displacement(es.positions[si3[i]:si3[i]+1], es.positions[si3[j]:si3[j]+1], cell)
        d = np.linalg.norm(dr)
        print(f"  d(Si{si3[i]} - Si{si3[j]}) = {d:.4f} A")
