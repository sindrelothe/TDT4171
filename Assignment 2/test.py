import numpy as np

a = np.array([0.818, 0.182])
b = np.array([0.69, 0.41])

c = a * b / sum(a * b)

print(c)
