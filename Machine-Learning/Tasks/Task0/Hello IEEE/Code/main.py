from matsum import matsum
from matsub import matsub
from matmul import matmul
from scalarsum import scalarsum
from scalarsub import scalarsub
from matnorm import matnorm

A = [[1,2,3],
     [4,5,6],
     [7,8,9]]
B = [[1,2,3],
     [4,5,6],
     [7,8,9]]

print("="*50)
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("="*50)
print("Matrix Sum:")
print(matsum(A,B))
print("="*50)
print("Matrix Sub:")
print(matsub(A,B))
print("="*50)
print("Matrix Mul:")
print(matmul(A,B))
print("="*50)
print("Scalar Sum:")
print(scalarsum(2,A))
print("="*50)
print("Scalar Sub:")
print(scalarsub(2,A))
print("="*50)
print("Matrix Norm:")
print(matnorm(A))
print("="*50)
