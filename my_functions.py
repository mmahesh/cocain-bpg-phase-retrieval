import numpy as np
np.random.seed(0)

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.image import imread



def main_func(A,b, U, lam, fun_num=1):
	if fun_num==1:
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = 0.25*(U.T.dot(item.dot(U)) - b[count])**2
			count +=1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(np.sum(np.abs(U))) 

		return temp_sum

	if fun_num==2:
		temp_sum = 0
		count = 0
		for item in A:
			temp_ent = 0.25*(U.T.dot(item.dot(U)) - b[count])**2
			count +=1
			temp_sum = temp_sum + temp_ent
		temp_sum = temp_sum + lam*(U.T.dot(U))

		return temp_sum

	

def grad(A,b, U,  lam, fun_num=1):
	if fun_num==1:
		temp_grad = 0
		count = 0
		for item in A:
			temp_grad = temp_grad + (U.T.dot(item.dot(U)) - b[count])*(item.dot(U))
			count +=1

		return temp_grad

	if fun_num == 2:
		temp_grad = 0
		count = 0
		for item in A:
			temp_grad = temp_grad + (U.T.dot(item.dot(U)) - b[count])*(item.dot(U))
			count +=1
		return temp_grad 

	

def abs_func(A,b, U, U1, lam, abs_fun_num=1, fun_num=1):
	if abs_fun_num == 1:
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return main_func(A,b, U1, lam, fun_num=fun_num)\
							+ np.sum(np.multiply(U-U1,G)) \
							-lam*(np.sum(np.abs(U1))) + lam*(np.sum(np.abs(U)))
	if abs_fun_num == 2:
		G = grad(A,b, U1, lam, fun_num=fun_num)
		return main_func(A,b, U1, lam, fun_num=fun_num) \
				+ np.sum(np.multiply(U-U1,G))-lam*(U1.T.dot(U1))+lam*(U.T.dot(U))
	
def breg( U, U1, breg_num=1, A=1,b=1, lam=1):
	if breg_num==1:
		grad_U1 = (np.sum(np.multiply(U1,U1)))*U1 + U1
		temp =  0.25*(np.linalg.norm(U)**4) + 0.5*(np.linalg.norm(U)**2) \
				- 0.25*(np.linalg.norm(U1)**4) - 0.5*(np.linalg.norm(U1)**2)\
				- np.sum(np.multiply(U-U1,grad_U1))
		if temp >=1e-15:
			return temp
		else:
			return 0
