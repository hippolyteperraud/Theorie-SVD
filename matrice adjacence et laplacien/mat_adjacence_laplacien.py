import numpy as np

# Lecture des arêtes, lignes vides ignorées
aretes = []
with open("aretes.txt") as f:
    for ligne in f:
        if ligne.strip():
            u, v = ligne.split(",")
            aretes.append((int(u), int(v)))

# Matrice d'adjacence
n = max(max(a) for a in aretes)
A = np.zeros((n, n), dtype=int)
for u, v in aretes:
    A[u - 1, v - 1] += 1
    A[v - 1, u - 1] += 1

# Matrice des degrés et laplacien
D = np.diag(A.sum(axis=1))
L = D - A

print("A =\n", A)
print("L =\n", L)