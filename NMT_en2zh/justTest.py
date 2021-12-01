import numpy as np

a = np.arange(50)[:, np.newaxis]
b = np.arange(512)[np.newaxis, :]
print(a.shape)
print(b.shape)

c = np.arange(10)[:, np.newaxis]
print(c.shape)
print(c//2)

d = np.reshape(np.arange(10), (10, 1))
print(d)
e = np.reshape(np.arange(8), (1, 8))
print(e)
print(d * e)