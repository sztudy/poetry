# encoding=utf-8

import numpy as np
import pickle

X = np.random.rand(100**2).reshape(100,100)
X = np.triu(X)
X += X.T - np.diag(X.diagonal())

#print(X == X.T)

values, vectors = np.linalg.eig(X)

#print(vectors[1])

print(np.dot(vectors.T[1], vectors.T[22]))

#print(vectors.T[1])

with open('model/vectors.pkl','wb') as f:
    pickle.dump([vectors.T[1], vectors.T[22]], f)

with open('model/vectors.pkl','rb') as f:
    vec = pickle.load(f)
print(vec)
