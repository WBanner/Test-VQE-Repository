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

def find_commutator(op_2, kwargs):
    op_1 = kwargs['op_1']
    return op_1*op_2 - op_2*op_1
def does_commute(op_2, kwargs):
    op_1 = kwargs['op_1']
    return op_2.commute_with(op_1)
def split_into_paulis(ham):
    ham_details = ham.print_details()
    ham_list = ham_details.split('\n')
    pauli_list = [0]*(len(ham_list)) #exclude last entry of ham list bc is just blank
    name_list = [0]*(len(ham_list))
    weight_list = [0]*(len(ham_list))
    for counter in range(0, len(ham_list),1):
        if ham_list[counter][5:-1] == '0':
        	pauli = Pauli.from_label(ham_list[counter][:4])
        	name_list[counter] = WeightedPauliOperator.from_list([pauli]).print_details()
        	weight_list[counter] = complex(0)
       		pauli_list[counter] = WeightedPauliOperator.from_list([pauli])
        else:
	        if ham_list[counter][6:-1] == '':
	            break
	        pauli = Pauli.from_label(ham_list[counter][:4])
	        name_list[counter] = WeightedPauliOperator.from_list([pauli]).print_details()
	        weight_list[counter] = complex(ham_list[counter][6:-1])
	        pauli_list[counter] = WeightedPauliOperator.from_list([pauli])
    return weight_list, pauli_list, name_list


class CADAPTVQE():
	def __init__(self, operator_pool: OperatorPool, hamiltonian:BaseOperator, optimizer, max_iters = 2):
		expecs = pd.read_csv('Pauli_values.csv')
		expecs = expecs.to_dict()
		self.num_qubits = hamiltonian.num_qubits
		self.max_iters = max_iters
		self.expecs = expecs
		self._operator_pool = operator_pool
		self.ham_weight_list, self.ham_pauli_list, self.ham_name_list = split_into_paulis(hamiltonian)
		self.energy_list = []
		self.op_list = []
		self.param_list = []
		self.hamiltonian = hamiltonian
		self._optimizer = optimizer
	def run(self):
		iters = 0
		identity = Pauli.from_label('I'*self.num_qubits)
		H_c_eng, H_a_eng, comm_eng = self._reconstruct_single_energy_expression(op = WeightedPauliOperator.from_list([identity]), ham = self.hamiltonian, op_finding = True)
		self.energy_list = [np.real(self._return_single_energy(param = 0, H_c = H_c_eng, H_a = H_a_eng, comm = comm_eng))]
		for iters in range(0,self.max_iters):
			next_op_info = self._find_next_operator()
			self.op_list = next_op_info[0]
			self.param_list = next_op_info[1]
			if iters > 0:
				self.energy_list.append(self._get_optimal_energy(iters)['min_val'])
			else:
				self.energy_list.append(next_op_info[2])
		return self.energy_list
	def _find_next_operator(self):
		A_list = []
		B_list = []
		C_list = []
		for op in self._operator_pool.pool:
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

	def _reconstruct_single_energy_expression(self, op, ham = None, op_finding = False):
		if op_finding == True:
			ham_pauli_list = self.ham_pauli_list
			ham_weight_list = self.ham_weight_list
			ham_name_list = self.ham_name_list
		else:
			ham_weight_list, ham_pauli_list, ham_name_list = split_into_paulis(ham)
		kwargs = {'op_1': op}
		args = []
		H_c_eng = 0
		H_a_eng = 0
		H_a_op = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)])
		H_c_op = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)])
		for i in range(0,(len(ham_pauli_list))):
			for j in range(0, (len(self.expecs['exp vals']))):
				if ham_name_list[i] == self.expecs['names'][j]:
					if does_commute(ham_pauli_list[i], kwargs):
						H_c_eng = H_c_eng + ham_weight_list[i]*complex(self.expecs['exp vals'][j])
						H_c_op = ham_pauli_list[i]*ham_weight_list[i] + H_c_op
					else:
						H_a_eng = H_a_eng + ham_weight_list[i]*complex(self.expecs['exp vals'][j])
						H_a_op = H_a_op + ham_pauli_list[i]*ham_weight_list[i]
		comm = find_commutator(H_a_op, kwargs) + 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)])
		comm_eng = self._find_constants(comm)
		if op_finding == True:
			return H_c_eng, H_a_eng, comm_eng
		else:
			return [H_c_op, H_a_op, comm]

	def _reconstruct_multi_energy_expression(self, ansz_length):
		Ham_list = [self.hamiltonian]
		for i in range(0,ansz_length):
			if i == 0:
				lower = 0
				upper = 1
			else:
				lower = int(3**(i-1) + lower)
				upper = int(lower + 3**i)
			for k in range(lower,upper):
				Ham_info = self._reconstruct_single_energy_expression(self.op_list[i], ham = Ham_list[k])
				Ham_list.append(Ham_info[0])
				Ham_list.append(Ham_info[1])
				Ham_list.append(Ham_info[2])
			if i == ansz_length-1:
				lower = int(3**(i) + lower)
				upper = int(lower + 3**(i+1))
				Eng_list = [0]*(upper - lower)
				for k in range(lower,upper):
					Eng_list[k-lower] = self._find_constants(Ham_list[k])
		return Eng_list
	def _find_constants(self, ham):
		Ham_eng = 0
		ham_weight_list, ham_pauli_list, ham_name_list = split_into_paulis(ham)
		for i in range(0,(len(ham_pauli_list))):
			for j in range(0, (len(self.expecs['exp vals']))):
				if ham_name_list[i] == self.expecs['names'][j]:
					Ham_eng = Ham_eng + ham_weight_list[i]*complex(self.expecs['exp vals'][j])
		return Ham_eng
	def _get_optimal_energy(self, iters):
		num_params = iters + 1
		self.expression_list = self._reconstruct_multi_energy_expression(num_params)
		ret = self.find_minimum(initial_point = self.param_list, cost_fn = self._return_multi_energy, optimizer = self._optimizer)
		return ret
	def _return_multi_energy(self, params):
		expression_list = self.expression_list
		Eng_val = expression_list
		for k in range(0,int(math.log(len(expression_list),3))):
			Eng_val_new = [0]*int(len(Eng_val)/3)
			for i in range(0, 3*len(Eng_val_new),3):
				Eng_val_new[int(i/3)] = self._return_single_energy(params[-(k+1)],Eng_val[i],Eng_val[i+1],Eng_val[i+2])
			Eng_val = Eng_val_new
		return float(Eng_val[0])

	def _return_single_energy(self, param, H_c, H_a, comm):
		energy = H_c + np.cos(param)*H_a + 1j*np.sin(param)*comm*0.5
		return energy

	def find_minimum(self, initial_point=None, var_form=None,
				cost_fn=None, optimizer=None, gradient_fn=None): 
		"""Optimize to find the minimum cost value.
		Returns:
			dict: Optimized variational parameters, and corresponding minimum cost value.

		Raises:
			ValueError: invalid input

		"""
		bounds=[(-np.pi, +np.pi)] * len(self.op_list)
		optimizer = self._optimizer
		nparms = len(self.param_list)

		if initial_point is not None and len(initial_point) != nparms: #error for parameter/initial point size mismatch
			raise ValueError(
				'Initial point size {} and parameter size {} mismatch'.format(
				len(initial_point), nparms))

		if initial_point is not None:
			if not optimizer.is_initial_point_supported:
				raise ValueError('Optimizer does not support initial point')
		else:
			if optimizer.is_initial_point_required: #sets new random initial points for optimizer within bound if initial pint required
				low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds] #can you redefine 1 to be NONE? I suppose so
				high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
				initial_point = self.random.uniform(low, high)

		start = time.time() #returns time as flt in seconds
		if not optimizer.is_gradient_supported:  # ignore the passed gradient function #if the optimizer isn't gradient based then grad = none
			gradient_fn = None

		opt_params, opt_val, num_optimizer_evals = optimizer.optimize(nparms, #here's the root optimization algorithm, optimize function does: whatever you define based on optimizer object class (optimizer is just base class with abstract optimize func)
																		cost_fn,
																		variable_bounds=bounds,
																		initial_point=initial_point,
																		gradient_function=gradient_fn)
		eval_time = time.time() - start #measures evaluation time.
		ret = {}
		ret['num_optimizer_evals'] = num_optimizer_evals
		ret['min_val'] = opt_val
		ret['opt_params'] = opt_params
		ret['eval_time'] = eval_time

		return ret #returns dictionary with number of evaluations, optimum cost value, optimum parameters, and evaluation time


