#Using CVXPY for Optimization Problem Solving
import cvxpy as cpy
#Using Numpy for solving equations
import numpy as npy
import scipy

#loading the data
C=npy.load("Data For Assignment RED/C.npy")
y=npy.load("Data For Assignment RED/y.npy")
A_inv=npy.load("Data For Assignment RED/A_inv.npy")

#Reshaping the size of y as a single column vector
y=y.reshape(y.shape[0],1)
#Transforming s matrix as it needs to be valid for matrix multiplication
s= cpy.Variable((C.shape[1],1))

#Using CVXPY library for solving convex optimzation problem
objective = cpy.Minimize(cpy.norm(s,1)); # objective function definition
#constraint = [(y-C@s)**2==0] #given constraint
constraint = [(y-C@s)==0] #this is similar as above constraint
prob=cpy.Problem(objective,constraint) #Posed the problem

result = prob.solve(solver=cpy.OSQP,verbose=True) 
# result i:e an optimal solution (column vector will be ovtained) verbose=True helped track the steps while the prob.solve was running

#Solving the linear algebra equation to obtain x.
x=scipy.linalg.solve(A_inv,s.value)#using the s

print(x)
print(s.value)

#contents were saved
npy.save("ans_s_red.npy",s)
npy.save("ans_x_red.npy",x)




