import numpy as np

A = np.array([[1, 1], [0.2, -1]])
Y = np.array([[1], [0]])
A_inverse = np.linalg.inv(A)
X = np.matmul(A_inverse, Y)
print("A inverse:")
print(A_inverse)
print("X:")
print(X)