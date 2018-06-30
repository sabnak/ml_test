import numpy as np
from scipy.sparse import csr_matrix
import sys

# print(sys.path)
#
#
# X = np.array([
# 	[0, 1, 0, 1],
# 	[1, 0, 1, 1],
# 	[0, 0, 0, 1],
# 	[1, 0, 1, 0]
# ])
# y = np.array([0, 1, 0, 1])
#
# print(X[y == 0])
#
# t = np.random.randint(1, 100, (5, 5))
#
#
# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# print(csr_matrix((data, (row, col)), shape=(3, 3)).toarray())


x = np.array([[[0, 3], [1, 3], [2, 3]]])
print(x)

print(np.squeeze(x))

