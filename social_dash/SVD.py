# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:20:02 2018

@author: Jie.Hu
"""



# Singular-value decomposition
from numpy import array
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)


# SVD
U, s, VT = svd(A)
print(U)
print(s)
print(VT)




# Reconstruct SVD
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[1], :A.shape[1]] = diag(s)
# reconstruct matrix
B = U.dot(Sigma.dot(VT))
print(B)

C = (U.dot(Sigma)).dot(VT)
print(C)




# Dimensionality Reduction
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
# define a matrix
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print(A)
# Singular-value decomposition
U, s, VT = svd(A)
# create m x n Sigma matrix
Sigma = zeros((A.shape[0], A.shape[1]))
# populate Sigma with n x n diagonal matrix
Sigma[:A.shape[0], :A.shape[0]] = diag(s)
# reconstruct matrix
C = (U.dot(Sigma)).dot(VT)
print(C)

# select
n_elements = 2
Sigma = Sigma[:, :n_elements]
VT = VT[:n_elements, :]
# reconstruct
B = U.dot(Sigma.dot(VT))
print(B)
# transform
T = U.dot(Sigma)
print(T)
T = A.dot(VT.T)
print(T)



from numpy import array
from sklearn.decomposition import TruncatedSVD
# define array
A = array([
	[1,2,3,4,5,6,7,8,9,10],
	[11,12,13,14,15,16,17,18,19,20],
	[21,22,23,24,25,26,27,28,29,30]])
print(A)
# svd
svd = TruncatedSVD(n_components=2)
svd.fit(A)
result = svd.transform(A)
print(result)

B = svd.inverse_transform(result)

C = array([
	[31,32,53,44,45,66,77,48,29,210],
	[11,12,13,134,15,16,17,18,19,20],
	[21,22,23,24,25,126,27,28,29,30]])
print(C) 

result = svd.transform(C)
print(result)

B = svd.inverse_transform(result)

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
# criterion 1.00/100=0.01
svd = TruncatedSVD(n_components=30, n_iter=7, algorithm='randomized', random_state=1234)
svd.fit(X)  

print(svd.explained_variance_ratio_)  

print(svd.explained_variance_ratio_.sum())  

print(svd.singular_values_)  

result = svd.transform(X)
print(result)

Y = sparse_random_matrix(20, 100, density=0.5, random_state=42)

result2 = svd.transform(Y)
print(result2)