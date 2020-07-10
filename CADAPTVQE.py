import numpy as np
import pandas as pd
import scipy
import math
from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool
import time
import logging

logger = logging.getLogger(__name__)



class CADAPTVQE():
	def __init__(self,
	  operator_pool,
	  hamiltonian,
	  optimizer, 
	  max_iters, 
	  operator_selector, 
	  expecs,
	  expec_mode = False,
	  initial_state = 0,
	  grad_tol,
	  use_zero_initial_parameters):
	self.op_pool = operator_pool
	self.ham = hamiltonian
	self.optimizer = optimizer
	self.max_iters = max_iters
	self.grad_tol = grad_tol
	self.use_zero_initial_parameters = use_zero_initial_parameters
	self.operator_selector = operator_selector
	self.num_qubits = hamiltonian.num_qubits
	self.energy_list = []
	self.op_list = []
	self.expec_mode = 
	self.param_list = []


	def add_initial_point(self, num_params):
		init_list = []
		if self.use_zero_initial_parameters:
			init_list = [0]*num_params
		else:
			for i in range(0,len(num_params)):
				init_list.append(np.random.uniform(0,np.pi))
		return init_list



	def run(self):
		iters = 0
		initial_point = add_initial_point(1)
		initial_cvqe = CVQE(operator = self.hamiltonian, ham_list = [self.hamiltonian], var_form = None, optimizer = self._optimizer, expecs = self.expecs, initial_point = [0])
		ret, term_list, base_3_list, energy_list = CVQE.run()
		self.energy_list = [ret['min_val']]
		
		while max_grad > self.grad_tol:
			next_op_param_info = self._find_next_operator_init_param()
			self.op_list = next_op_param_info[0]
			self.param_list = next_op_param_info[1]
			if iters > 0:
				self.energy_list.append(self._get_optimal_energy(iters)['min_val'])
			else:
				self.energy_list.append(next_op_info[2])
		return self.energy_list


	def _find_next_operator_init_param(self, initial_param = None):
		if initial_param == None:
			initial_param = 0
		grad_list
		potential_list
		potential_ham_list
		for i in number of ops in pool #need to parallelize
			cvqe = CVQE
			potential_ham_list = CVQE._reconstruct_multi_energy_expression()
			potential_energy_list = CVQE._create_energy_list
			grad_list.append(CVQE.evaluate_multi_energy())
		find max of energy_list
		find corresponding operator
		

		#now to implement adapt

		return [self.op_list + [next_op], self.param_list + [np.real(next_param)], np.real(Optim_energy_array[optim_param_pos])]


"""
	def _find_next_operator(self):
		A_list = []
		B_list = []
		C_list = []
		for op in self._operator_pool.pool:
		#bug- we need to reconstruct multi-energy expression here:
			A,B,C = self._reconstruct_single_energy_expression(op = op, ham = self.hamiltonian, op_finding = True)
			A_list.append(A)
			B_list.append(B)
			C_list.append(C)
		A = np.array(A_list)
		B = np.array(B_list)
		C = np.array(C_list)
		Amp = np.sqrt(np.square(B)-np.square(C)/4)
		Optim_energy_array = A - Amp
		bottom_arr = np.real(1j*C)
		top_arr = np.real(2*B)
		Optim_param_array = -np.arctan2(top_arr, bottom_arr) - np.pi/2
		optim_param_pos = np.argmin(Optim_energy_array)
		next_op = self._operator_pool.pool[optim_param_pos]
		next_param = Optim_param_array[optim_param_pos]
		return [self.op_list + [next_op], self.param_list + [np.real(next_param)], np.real(Optim_energy_array[optim_param_pos])]
"""
	