"""
Code for Phase Retrieval Experiments (Section 6.3) of the paper:
Convex-Concave Backtracking for Inertial Bregman Proximal Gradient 
Algorithms in Non-Convex Optimization
Paper authors: Mahesh Chandra Mukkamala, Peter Ochs, 
Thomas Pock and Shoham Sabach.

Other related codes at:
https://github.com/mmahesh/cocain-bpg-escapes-spurious-stationary-points
https://github.com/mmahesh/cocain-bpg-matrix-factorization

Algorithms Implemented:
BPG: Bregman Proximal Gradient
CoCaIn BPG: Convex Concave Inertial (CoCaIn) BPG 
BPG-WB: BPG with Backtracking
IBPM-LS: Inexact Bregman Proximal Minimization Line Search Algorithm

References:
CoCaIn BPG paper: https://arxiv.org/abs/1904.03537
BPG paper: https://arxiv.org/abs/1706.06461
IBPM-LS paper: https://arxiv.org/abs/1707.02278

Contact: Mahesh Chandra Mukkamala (mukkamala@math.uni-sb.de)
"""

import numpy as np

from my_functions import *

import argparse
parser = argparse.ArgumentParser(description='Phase Retrieval Experiments')
parser.add_argument('--lam', '--regularization-parameter',
					default=1e-1, type=float,  dest='lam')
parser.add_argument('--algo', '--algorithm', default=1, type=int,  dest='algo')
parser.add_argument('--max_iter', '--max_iter',
					default=1000, type=int,  dest='max_iter')
parser.add_argument('--fun_num', '--fun_num', default=1,
					type=int,  dest='fun_num')
parser.add_argument('--abs_fun_num', '--abs_fun_num',
					default=1, type=int,  dest='abs_fun_num')
parser.add_argument('--breg_num', '--breg_num',
					default=1, type=int,  dest='breg_num')
args = parser.parse_args()


# some backward compatibility and initialization
lam = args.lam
algo = args.algo
fun_num = args.fun_num
abs_fun_num = args.abs_fun_num
breg_num = args.breg_num

# for forward backward splitting only
np.random.seed(0)

max_iter = 1000

dim = 10  # dimension 10
temp_Alist = []
temp_blist = []

global_L = 0

for i in range(100):
	temp_A = np.random.rand(dim, 1)
	A = temp_A*temp_A.T
	temp_Alist = temp_Alist + [A]
	temp_b = np.random.rand(1)[0]
	global_L = global_L + 3*(np.linalg.norm(A)**2) + \
		np.linalg.norm(A)*abs(temp_b)
	temp_blist = temp_blist + [temp_b]


A = temp_Alist
b = temp_blist

if fun_num == 1:
	# fun_num = 1 for L1 Regularization
	del_val = 0.15
	eps_val = 0.00001
	uL_est =  10
	lL_est = 1e-4*uL_est
	init_U = uL_est
	U = np.ones(dim)
	lam = 1e-1

	prev_U = U


if fun_num == 2:
	# fun_num = 2 for L2 Regularization
	del_val = 0.15
	eps_val = 0.00001
	uL_est = 10
	lL_est = 1e-4*uL_est
	U = np.ones(dim)
	init_U = uL_est
	lam = 1e-1

	prev_U = U

lL_est_main = lL_est


def make_update(y, grad, uL_est):
	# Bregman Proximal Mapping with L1 Regularization

	temp_p = (1/uL_est)*grad - (y.T.dot(y))*y - y
	temp_p = np.maximum(0, np.abs(temp_p)-lam*(1/uL_est))*np.sign(-temp_p)
	temp_pnorm = np.linalg.norm(temp_p)**2
	coeff = [temp_pnorm, 0, 1, -1]
	temp_y = np.roots(coeff)[-1].real
	print('temp_y L1 is ' + str(temp_y))
	return temp_y*temp_p


def make_update1(y, grad, uL_est):
	# Bregman Proximal Mapping with L2 Regularization

	temp_p = (1/uL_est)*grad - (y.T.dot(y))*y - y
	temp_pnorm = np.linalg.norm(temp_p)**2
	coeff = [temp_pnorm, 0, (2*lam*(1/uL_est) + 1), 1]
	temp_y = np.roots(coeff)[-1].real
	print('temp_y L2 is '+ str(temp_y))
	return temp_y*temp_p


def find_gamma(A, b, U, prev_U, uL_est, lL_est):
	# Finding the inertial parameter gamma

	gamma = 1
	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est))
	y_U = U + gamma*(U-prev_U)
	while (kappa*breg(prev_U, U, breg_num=breg_num, A=A, b=b, lam=lam) \
		< breg(U, y_U, breg_num=breg_num,  A=A, b=b, lam=lam)):
		gamma = gamma*0.9
		y_U = U + gamma*(U-prev_U)
	return y_U, gamma


def find_closed_gamma(A, b, U, prev_U, uL_est, lL_est):
	# Finding the inertial parameter gamma

	kappa = (del_val - eps_val)*(uL_est/(uL_est+lL_est))
	

	Delta_val = np.linalg.norm(U-prev_U)**2
	print(Delta_val)
	if Delta_val <0:
		y_U = U
		gamma = 0
	else:
		temp_var = (1.5*Delta_val + (7/4) )*(np.linalg.norm(U)**2)
		gamma = np.sqrt(kappa*breg(prev_U, U,\
				 breg_num=breg_num, A=A, b=b, lam=lam)/temp_var)
		y_U = U + gamma*(U-prev_U)
	return y_U, gamma


def do_lb_search(A, b, U, U1, lam, uL_est, lL_est, closed_form=0):
	# Lower bound backtracking

	if closed_form==0:
		y_U, gamma = find_gamma(A, b, U, U1, uL_est, lL_est)
	else:
		y_U, gamma = find_closed_gamma(A, b, U, U1, uL_est, lL_est)


	while((abs_func(A, b, U, y_U, lam, abs_fun_num=abs_fun_num, fun_num=fun_num)
			- main_func(A, b, U, lam, fun_num=fun_num)
			- (lL_est*breg(U, y_U,breg_num=breg_num, A=A, b=b, lam=lam))) > 1e-7):

		lL_est = (2)*lL_est
		if closed_form==0:
			y_U, gamma = find_gamma(A, b, U, U1, uL_est, lL_est)
		else:
			y_U, gamma = find_closed_gamma(A, b, U, U1, uL_est, lL_est)

	return lL_est, y_U, gamma


def do_ub_search(A, b, y_U, uL_est):

	# compute gradients
	grad_u = grad(A, b, y_U, lam, fun_num=fun_num)

	# make update step
	if fun_num == 1:
		x_U = make_update(y_U, grad_u, uL_est)
	elif fun_num == 2:
		x_U = make_update1(y_U, grad_u, uL_est)
	else:
		raise

	
	delta_new = (abs_func(A, b, x_U, y_U, lam, abs_fun_num=abs_fun_num, fun_num=fun_num)
               - main_func(A, b, x_U, lam,  fun_num=fun_num)
               + (uL_est*breg(x_U, y_U, breg_num=breg_num, A=A, b=b, lam=lam)))
	print('Delta is ' + str(delta_new))
	while((delta_new < -1e-7)) :
		
		delta_prev = delta_new
		delta_new = (abs_func(A, b, x_U, y_U, lam, abs_fun_num=abs_fun_num, 	fun_num=fun_num)
                    - main_func(A, b, x_U, lam,  fun_num=fun_num)
                    + (uL_est*breg(x_U, y_U, breg_num=breg_num, A=A, b=b, lam=lam)))

		print('Delta is '+ str(delta_new))
		
		uL_est = (2)*uL_est
		print('uL_est is '+ str(uL_est))
		# make update step
		if fun_num == 1:
			x_U = make_update(y_U, grad_u, uL_est)
		elif fun_num == 2:
			x_U = make_update1(y_U, grad_u, uL_est)
		else:
			raise

	return uL_est, x_U


def obtain_delta(A, b, y_U, uL_est):
	grad_u = grad(A, b, y_U, lam, fun_num=fun_num)
	if fun_num == 1:
		tx_U = make_update(y_U, grad_u, uL_est)
	elif fun_num == 2:
		tx_U = make_update1(y_U, grad_u, uL_est)
	else:
		raise
	temp_delta = (abs_func(A, b, tx_U, y_U, lam, \
						abs_fun_num=abs_fun_num, fun_num=fun_num)
				  - main_func(A, b, y_U, lam,  fun_num=fun_num)
				  + (uL_est*breg(tx_U, y_U, breg_num=breg_num)))
	return temp_delta, tx_U


def line_search(y_U):
	gm = 0.001
	eta = 0.001
	# here some gm, eta values can be unstable towards the end
	# requires some tuning
	# the above values work fine

	delta, tx_U = obtain_delta(A, b, y_U, uL_est)
	x_U = y_U + eta*(tx_U - y_U)

	while(main_func(A, b, x_U, lam,  fun_num=fun_num) \
		- main_func(A, b, y_U, lam,  fun_num=fun_num) \
		- (eta*gm*delta) > 1e-7) and (delta > 0):
		eta = eta*0.1
		x_U = y_U + eta*(tx_U - y_U)

	return x_U


if algo == 1:
	# Implementation of CoCaIn BPG
	gamma_vals = [0]
	uL_est_vals = [uL_est]
	lL_est_vals = [lL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	lyapunov_vals = [temp]

	U_vals = [init_U]
	# U2_vals = []
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		lL_est, y_U, gamma = do_lb_search(
			A, b, U, prev_U,  lam, uL_est, lL_est=lL_est_main)
		prev_U = U
		print("doing Lb " + str(lL_est))
		print("doing Ub " + str(uL_est))
		temp_ulest = uL_est

		uL_est, U = do_ub_search(A, b, y_U, uL_est)
		# print('funct value at '+ str(i) + ' is ')
		print(main_func(A, b, U, lam, fun_num=fun_num))
		uL_est_vals = uL_est_vals + [uL_est]
		lL_est_vals = lL_est_vals + [lL_est]

		gamma_vals = gamma_vals + [gamma]
		U_vals = U_vals + [U]
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		lyapunov_vals = lyapunov_vals + \
			[(1/uL_est)*temp+breg(U, prev_U, breg_num=breg_num, A=A, b=b, lam=lam)]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/cocain_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, lyapunov_vals,
							   uL_est_vals, lL_est_vals, gamma_vals, time_vals])
if algo == 2:
	# Implementation of BPG with Backtracking
	uL_est_vals = [uL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	prev_fun_val = temp
	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0

	for i in range(max_iter):
		st_time = time.time()

		uL_est, U = do_ub_search(A, b, U, uL_est)
		print(main_func(A, b, U, lam, fun_num=fun_num))
		print('uL_est is '+ str(uL_est))
		uL_est_vals = uL_est_vals + [uL_est]

		temp = main_func(A, b, U, lam, fun_num=fun_num)
		#print('fun val is '+ str(temp))
		if np.isnan(temp):
			raise
		if temp >prev_fun_val:
			print('Function value increases')
			raise
		prev_fun_val = temp
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/gd_bt_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals,  time_vals, uL_est_vals])
if algo == 3:
	# Implementation of BPG without backtracking
	# Here global_L governs the step-size

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		gamma = 0

		uL_est = global_L
		grad_u = grad(A, b, U, lam, fun_num=fun_num)
		if fun_num == 1:
			U = make_update(U, grad_u, uL_est)
		elif fun_num == 2:
			U = make_update1(U, grad_u, uL_est)
		else:
			raise

		print(main_func(A, b, U, lam, fun_num=fun_num))

		temp = main_func(A, b, U, lam, fun_num=fun_num)
		#print('fun val is '+ str(temp))
		if np.isnan(temp):
			raise
		
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time
	filename = 'results/gd_bt_global_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, time_vals])
if algo == 4:
	# IBPM-LS from https://arxiv.org/abs/1707.02278

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		gamma = 0
		U = line_search(U)
		print(main_func(A, b, U, lam, fun_num=fun_num))
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/ibgm_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, time_vals])

if algo == 5:
	# Implementation of CoCaIn BPG with Closed form Inertia
	gamma_vals = [0]
	uL_est_vals = [uL_est]
	lL_est_vals = [lL_est]

	temp = main_func(A, b, U, lam, fun_num=fun_num)
	print('temp is ' + str(temp))

	func_vals = [temp]
	lyapunov_vals = [temp]

	U_vals = [init_U]
	# U2_vals = []
	import time
	time_vals = np.zeros(max_iter+1)
	time_vals[0] = 0
	for i in range(max_iter):
		st_time = time.time()
		lL_est, y_U, gamma = do_lb_search(
			A, b, U, prev_U,  lam, uL_est, lL_est=lL_est_main, closed_form=1)
		prev_U = U
		print("doing Lb " + str(lL_est))
		print("doing Ub " + str(uL_est))
		temp_ulest = uL_est

		uL_est, U = do_ub_search(A, b, y_U, uL_est)
		# print('funct value at '+ str(i) + ' is ')
		print(main_func(A, b, U, lam, fun_num=fun_num))
		uL_est_vals = uL_est_vals + [uL_est]
		lL_est_vals = lL_est_vals + [lL_est]

		gamma_vals = gamma_vals + [gamma]
		U_vals = U_vals + [U]
		temp = main_func(A, b, U, lam, fun_num=fun_num)
		if np.isnan(temp):
			raise
		func_vals = func_vals + [temp]
		lyapunov_vals = lyapunov_vals + \
			[(1/uL_est)*temp+breg(U, prev_U, breg_num=breg_num, A=A, b=b, lam=lam)]
		time_vals[i+1] = time.time() - st_time

	filename = 'results/cocain_cf_' + \
		str(fun_num)+'_abs_fun_num_'+str(abs_fun_num)+'.txt'
	np.savetxt(filename, np.c_[func_vals, lyapunov_vals,
							   uL_est_vals, lL_est_vals, gamma_vals, time_vals])
