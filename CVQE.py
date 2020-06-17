import numpy as np
import pandas as pd
import math
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qisresearch.adapt.operator_pool import PauliPool
from qiskit.aqua.algorithms import VQE
import time

def find_commutator(op_1, kwargs):
	"""
		finds the commutator of 2 operators
	"""
	op_2 = kwargs['op_2']
	return op_1*op_2 - op_2*op_1

def does_commute(op_1, kwargs):
	"""
		checks if operators commute
	"""
	op_2 = kwargs['op_2']
	return op_1.commute_with(op_2)

def split_into_paulis(ham):
	"""
		Used to split the given hamiltonian into lists of constituent parts:
		weight_list: list of complex floats that are the weights of terms in the hamiltonian
		pauli_list: a list of the Weighted_pauli_operator representations of the pauli strings in the hamiltonian
					(does not include weights)
		name_list: a list of strings that correspond to the names of the operators in the hamiltonian
	"""
	num_qubits = ham.num_qubits
	ham_details = ham.print_details()
	ham_list = ham_details.split('\n')
	pauli_list = [0]*(len(ham_list)-1) #exclude last entry of ham list bc is just blank
	name_list = [0]*(len(ham_list)-1)
	weight_list = [0]*(len(ham_list)-1)
	if len(ham_list) > 2:
		for counter in range(0, len(ham_list),1):
			if ham_list[counter][num_qubits + 2:] == '':
				break
			if ham_list[counter][num_qubits + 1:] == '0':
				name_list[counter] = 'I'*num_qubits
				weight_list[counter] = complex(0)
				pauli_list[counter] = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
			else:
				pauli = Pauli.from_label(ham_list[counter][:num_qubits])
				name_list[counter] = ham_list[counter][:num_qubits]
				weight_list[counter] = complex(ham.paulis[counter][0])
				pauli_list[counter] = WeightedPauliOperator.from_list([pauli])
	else:
			if ham_list[0][num_qubits + 1:-1] == '0':
				name_list[0] = 'I'*num_qubits
				weight_list[0] = complex(0)
				pauli_list[0] = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
			else:
				pauli = Pauli.from_label(ham_list[0][:num_qubits])
				name_list[0] = ham_list[0][:num_qubits]
				weight_list[0] = complex(ham.paulis[0][0])
				pauli_list[0] = WeightedPauliOperator.from_list([pauli])
	return weight_list, pauli_list, name_list

def base_4_map(name):
	"""
		Used to convert a pauli string (essentially base 4) into a base 10 number for lookup in the expec value table
	"""
	entry = 0
	for i in range(0,len(name)):
		if name[len(name) - i - 1] == 'I':
			entry = entry + 0
		if name[len(name) - i - 1] == 'X':
			entry = entry + 4**i
		if name[len(name) - i - 1] == 'Y':
			entry = entry + 2*4**i
		if name[len(name) - i - 1] == 'Z':
			entry = entry + 3*4**i
	return entry

def create_num_list(pauli_name_list):
	num_qubits = len(pauli_name_list[0])
	num_list = []
	for i in range(0,len(pauli_name_list)):
		num_list.append(base_4_map(pauli_name_list[i]))
	return num_list

def ternary (n): #this function taken from stackexchange
	"""
		converts any base 10 number to base 3
		Is used in this case to convert the "nonzero term list" entry to base 3
		as a base 3 representation can be used to reconstruct the sine and cosine constant multiples
	"""
	if n == 0:
		return '0'
	nums = []
	while n:
		n, r = divmod(n, 3)
		nums.append(str(r))
	return ''.join(reversed(nums))

class Classical_VQE(VQE):

	def __init__(self, 
		operator, 
		optimizer, 
		var_form = None, 
		max_evals_grouped = 1, 
		aux_operators = None, 
		callback = None, 
		auto_conversion = True, 
		initial_point = None, 
		ham_list = None, 
		operator_list = None, 
		nonzero_terms = None, 
		starting_point = False,
		expecs = None):


		super().__init__(
			operator = operator,
			var_form = var_form,
			optimizer = optimizer,
			initial_point = initial_point,
			max_evals_grouped = max_evals_grouped,
			aux_operators = aux_operators,
			callback = callback,
			auto_conversion = auto_conversion
			)
		self.num_qubits = operator.num_qubits
		if expecs == None:
			expecs = pd.read_csv('{}_qubit_pauli_values.csv'.format(self.num_qubits)) #this file is required unless you pass in the expecation values (in order) manually
			expecs = expecs.to_dict()
		self.expecs = expecs
		if operator_list is not None:
			self.op_list = operator_list
		else:
			self.op_list = var_form.operator_pool.pool
		if initial_point is not None:
			self.param_list = initial_point
		else:
			self.param_list = [0]*len(self.op_list)
		self.starting_point = starting_point
		if starting_point:
			self.ham_list = ham_list
			self.nonzero_terms = nonzero_terms
		self.zero_term = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)])
		self.empt_term = self.zero_term - self.zero_term

	def run(self):
		"""
			runs a VQE classically
		"""
		if self.starting_point == False:
			startime = time.time()
			self.ham_list, self.nonzero_terms, for_time_sum = self._reconstruct_multi_energy_expression()
			construct_time = time.time()-startime
			print('construct time', construct_time)
		else:
			print('cant do this yet')
			return 0
		self.ham_energy_list = self._create_energy_list()
		if len(self.ham_energy_list) > 1:
			ret = self.find_minimum(initial_point = self.param_list, cost_fn = self._evaluate_multi_energy, optimizer = self._optimizer)
		else:
			ret = {'min_val': self.ham_energy_list[0]}
		return ret, for_time_sum

	def _create_energy_list(self):
		"""
			creates a list of expectation values for the terms in "ham_list"
		"""
		energy_list = []
		if len(self.ham_list) > 1:
			for ham in self.ham_list:
				energy_list.append(self._evaluate_single_energy(ham))
		else:
			energy_list.append(self._evaluate_single_energy(self.ham_list[0]))
		return energy_list

	def _evaluate_single_energy(self, ham):
		"""
			returns the expecation value of an operator based on the 
			pre-measured expecation values using the initial condition.
		"""
		ham_name_list = []
		for i in range(0,len(ham['p'])):
			ham_name_list.append(ham['p'][i].print_details()[0:self.num_qubits])
		Eng = 0
		for i in range(0,(len(ham['p']))):
			j = base_4_map(ham_name_list[i])
			if self.expecs['exp vals'][j] != complex(0):
				Eng = Eng + ham['w'][i]*complex(self.expecs['exp vals'][j])
		return Eng

	def _evaluate_multi_energy(self, param_list):
		"""
		This is the objective function to be optimized
			returns: Energy, the energy of the system at the given parameters
		"""
		ham_energy_list = self.ham_energy_list
		nonzero_terms = self.nonzero_terms
		Energy = 0
		for i,eng in enumerate(ham_energy_list):
			constant = 1
			base_3 = ternary(nonzero_terms[i]-1)
			for i in range(0,len(base_3)):
				if base_3[i] == '0':
					constant = constant
				elif base_3[i] == '1':
					constant = constant*np.cos(2*param_list[len(base_3) - i - 1])#currently not set to half angle to better compare with VQE that uses adapt variational form
				else:
					constant = -constant*0.5*1j*np.sin(2*param_list[len(base_3) - i - 1])
				if constant == 0:
					break
			Energy = constant*eng + Energy
		return np.real(Energy)

	def _reconstruct_single_energy_expression(self, op, ham = None):
		"""
			H_c: a weighted pauli operator of the terms in the passed hamiltonian that commute with the passed operator
			H_a: a weighted pauli operator of the terms in the passed hamiltonian that anticommute with the passed operator
			comm: a weighted pauli operator of the commutator [H_a, op]
		"""
		kwargs = {'op_2': op}
		H_a = {'w': [], 'p': []}
		H_c = {'w': [], 'p': []}
		comm = {'w': [], 'p': []}
		for_time = time.time()
		for i in range(0,(len(ham['p']))):
			if does_commute(ham['p'][i], kwargs):
				H_c['w'].append(ham['w'][i])
				H_c['p'].append(ham['p'][i])
			else:
				H_a['w'].append(ham['w'][i])
				H_a['p'].append(ham['p'][i])
				comm_op = ham['w'][i]*find_commutator(ham['p'][i], kwargs)
				comm_weight, comm_pauli, comm_name = split_into_paulis(comm_op)
				comm['w'].append(comm_weight[0])
				comm['p'].append(comm_pauli[0])
		if not H_c['w']:
			H_c['w'] = [0]
			H_c['p'] = [self.zero_term]
		if not H_a['w']:
			H_a['w'] = [0]
			H_a['p'] = [self.zero_term]
		if not comm['w']:
			comm['w'] = [0]
			comm['p'] = [self.zero_term]
		for_time = time.time() - for_time
		return H_c, H_a, comm, for_time

	def _reconstruct_multi_energy_expression(self):
		"""
			returns:
			ham_list: list of nonzero weighted pauli operators. is (typically) an exponentially (with ansatz length)
						sized list of separate terms. When measured and multiplied by sines and cosines, it can be used to evaluate the energy
			nonzero_term list: used to keep track of when terms in the ham_list correspond to which term in the overall energy expression, as empty
								or zero terms are not included in the ham_list.
		"""
		ham_weight_list, ham_pauli_list, ham_name_list = split_into_paulis(self._operator)
		ham_dict = {'w': ham_weight_list,'p': ham_pauli_list}
		ham_list = [ham_dict]
		nonzero_terms = [1]
		for_time_sum = 0
		for num, op in enumerate(self.op_list):
			num_prev_terms = len(ham_list)
			for i,ham in enumerate(ham_list[:num_prev_terms]):
				H_c, H_a, comm, for_time = self._reconstruct_single_energy_expression(op, ham)
				for_time_sum = for_time_sum + for_time
				if H_c['p'][0] != self.zero_term:
					ham_list.append(H_c)
					nonzero_terms.append(3*(nonzero_terms[i]-1) + 1)
				if H_a['p'][0] != self.zero_term:
					ham_list.append(H_a)
					nonzero_terms.append(3*(nonzero_terms[i]-1) + 2)
				if comm['p'][0] != self.zero_term:
					ham_list.append(comm)
					nonzero_terms.append(3*(nonzero_terms[i]-1) + 3)
			ham_list = ham_list[num_prev_terms:]
			nonzero_terms = nonzero_terms[num_prev_terms:]
		print('fortimesum', for_time_sum)
		return ham_list, nonzero_terms, for_time_sum