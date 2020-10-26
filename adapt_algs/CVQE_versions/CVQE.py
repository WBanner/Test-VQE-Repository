import numpy as np
import pandas as pd
import math
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qisresearch.adapt.operator_pool import PauliPool
from qiskit.aqua.algorithms import VQE
from qiskit.tools import parallel_map
import time
from copy import deepcopy
from qiskit.utils import aqua_globals

def reverse(lst): 
    lst.reverse() 
    return lst

def find_commutator(op_1, args, kwargs):
	"""
		finds the commutator of 2 operators
	"""
	op_2 = kwargs['op_2']
	return (op_1*op_2 - op_2*op_1)

def commute_yes(ham, args, kwargs):
	op_2 = kwargs['op_2']
	if ham.commute_with(op_2):
		return ham
	else:
		return

def commute_no(ham, args, kwargs):
	op_2 = kwargs['op_2']
	if not ham.commute_with(op_2):
		return ham
	else:
		return

def reconstruct_single_energy_expression(self, op, ham = None, zero_term):
		"""
			H_c: a weighted pauli operator of the terms in the passed hamiltonian that commute with the passed operator
			H_a: a weighted pauli operator of the terms in the passed hamiltonian that anticommute with the passed operator
			comm: a weighted pauli operator of the commutator [H_a, op]
		"""
		kwargs = {'op_2': op}
		args = []
		H_c = parallel_map(commute_yes, ham, args, kwargs)
		H_a = parallel_map(commute_no, ham, args, kwargs)
		comm = parallel_map(find_commutator,H_a,args,kwargs)

		if not H_c:
			H_c.append(zero_term)
		if not H_a:
			H_a.append(zero_term)
		return H_c, H_a, comm


def create_new_terms(term, zero_term, args):
	op = args[0]
	num = args[1]
	ham = term[0]
	term_number = term_list[1]
	new_term_list = []
	H_c, H_a, comm = reconstruct_single_energy_expression(op, ham, zero_term)
	if H_c[0] != zero_term:
		new_ham_list.append([H_c, 3*(term_number-1) + 1])
	if H_a[0] != zero_term:
		new_ham_list.append([H_a, 3*(term_number-1) + 2])
	if comm[0] != zero_term:
		new_ham_list.append([comm, 3*(term_number-1) + 3])


def reconstruct_multi_energy_expression(self, operator, op_list = None, tree = None):
		"""
			returns:
			ham_list: list of nonzero weighted pauli operators. is (typically) an exponentially (with ansatz length)
						sized list of separate terms. When measured and multiplied by sines and cosines, it can be used to evaluate the energy
			nonzero_term list: used to keep track of when terms in the ham_list correspond to which term in the overall energy expression, as empty
								or zero terms are not included in the ham_list.
		"""
		#can we take nonzero terms to tree list?
		ham = split_into_paulis(operator)
		term_list = [[ham, 1]]
		for num, op in enumerate(op_list):
			args = [op, num, self.zero_term]
			meta_term_list = parallel_map(create_new_term_list, term_list, args)
			new_term_list = []
			for term in meta_term_list:
				new_term_list = new_term_list + term
			if tree is not None:
				for term in new_term_list:
					i = 0
					if term[1] == 3*(term_list[i] - 1) + 1:
						tree[num + 1].append(0)
						i = i + 1
					if term[1] == 3*(term_list[i] - 1) + 2:
						tree[num + 1] = tree[num + 1] + [1,2]
						i = i+2

		if tree is not None:
			ham_list = parallel_map(return_first_entry, ham_list)
			return ham_list, tree
		else:
			return ham_list, nonzero_terms


def sort_term(term, previous_ops_list, op):
		if does_commute(term['term'], {'op_2': op}):
	else:
		descendants_list, term['tree'] = reconstruct_multi_energy_expression(find_commutator(term['term'], {'op_2': op}), previous_ops_list, term['tree'])
		term['descendants'] = term['descendants'] + descendants_list

def convert_to_wpauli_list(term, args):
		if term[0] == complex(0):
			separated_ham = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
		else:
			separated_ham = WeightedPauliOperator.from_list([term[1]],[term[0]])
	return separated_ham


def split_into_paulis(ham):
	"""
		Used to split the given hamiltonian into lists of constituent parts:
		weight_list: list of complex floats that are the weights of terms in the hamiltonian
		pauli_list: a list of the Weighted_pauli_operator representations of the pauli strings in the hamiltonian
					(does not include weights)
		name_list: a list of strings that correspond to the names of the operators in the hamiltonian
	"""
	ham_list = ham.paulis
	separated_ham_list = parallel_map(convert_to_wpauli_list, ham_list, args, num_processes = aqua_globals.num_processes)

	return separated_ham_list

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
def check_non_z(name):
	entry = 0
	for i in range(0,len(name)):
		if name[len(name) - i - 1] == 'I':
			entry = entry + 0
		if name[len(name) - i - 1] == 'X':
			entry = entry + 1
		if name[len(name) - i - 1] == 'Y':
			entry = entry + 1
		if name[len(name) - i - 1] == 'Z':
			entry = entry + 0
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


def generate_base_3_list_single_op(term)
	base_3_list = []
	two_counter = [0]*len(term['tree'])
	base_3_grid = reverse(term['tree'][:])
	for i in range(0,len(base_3_grid[0])):
		already_added_flag = 0
		pos = i
		for gen_num, generation in enumerate(base_3_grid):
			if gen_num == 0:
				base_3_list.append(str(generation[i]))
				if base_3_list[i][0] == '2':
					already_added_flag = 1
					two_counter[gen_num] = two_counter[gen_num] + 1
			else:
				pos = pos - two_counter[gen_num-1]
				base_3_list[i] = str(generation[pos]) + base_3_list[i]
				if base_3_list[i][0] == '2' and not already_added_flag:
					already_added_flag = 1
					two_counter[gen_num] = two_counter[gen_num] + 1


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
		expecs = None,
		expec_mode = False,
		dir_to_bracket = False,
		ham_term_list = None,
		ham_energy_list = None,
		base_3_list = None):


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
		if expecs == None and expec_mode == True:
			expecs = pd.read_csv('{}_qubit_pauli_values.csv'.format(self.num_qubits)) #this file is required unless you pass in the expecation values (in order) manually
			expecs = expecs.to_dict()
		if expec_mode == True:
			self.expecs = expecs
		else:
			self.expecs = []
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
		self.dir_to_bracket = dir_to_bracket
		self.ham_term_list = ham_term_list
		self.ham_energy_list = ham_energy_list
		self.base_3_list = base_3_list

	def run(self, return_lists = False):
		"""
			runs a VQE classically
		"""
		if self.starting_point == False and self.dir_to_bracket:
			self.ham_list, self.nonzero_terms = self._reconstruct_multi_energy_expression(self._operator, self.op_list)
		if self.starting_point == False and not self.dir_to_bracket:
			self.ham_term_list = self._reconstruct_op_descendants()
			self.base_3_list = self._generate_base_3_list()
		self.ham_energy_list = self._create_energy_list()
		if len(self.ham_energy_list) > 1:
			ret = self.find_minimum(initial_point = self.param_list, cost_fn = self._evaluate_multi_energy, optimizer = self._optimizer)
		else:
			ret = {'min_val': self.ham_energy_list[0], 'opt_params': [0]*len(self.op_list)}
		if return_lists:
			return ret, self.ham_term_list, self.base_3_list, self.ham_energy_list
		else:
			return ret

	def _create_energy_list(self):
		"""
			creates a list of expectation values for the terms in "ham_list"
		"""
		energy_list = []
		args = [self.num_qubits]
		if self.dir_to_bracket:
			if len(self.ham_list) > 1:
					energy_list = parallel_map(evaluate_single_energy, ham, args, num_processes = aqua_globals.num_processes)
			else:
				energy_list.append(self._evaluate_single_energy(self.ham_list[0]))
		else:
			for term in self.ham_term_list:
				for i in range(0,len(term['descendants'])):
					weight, pauli, name = split_into_paulis(term['descendants'][i])
					if self.expecs:
						j = base_4_map(name[0])
						energy_list.append(weight[0]*self.expecs['exp vals'][j])
					else:
						if not check_non_z(name[0]):
							energy_list.append(weight[0])
						else:
							energy_list.append(0)
		return energy_list

	def _evaluate_single_energy(self, ham, args):
		"""
			returns the expecation value of an operator based on the 
			pre-measured expecation values using the initial condition.
		"""
		num_qubits = args[0]
		ham_name_list = []
		for i in range(0,len(ham['p'])):
			ham_name_list.append(ham['p'][i].print_details()[0:self.num_qubits])
		Eng = 0
		if self.expecs:
			for i in range(0,(len(ham['p']))):
				j = base_4_map(ham_name_list[i])
				Eng = Eng + ham['w'][i]*complex(self.expecs['exp vals'][j])
		else:
			for i in range(0,(len(ham['p']))):
				if not check_non_z(ham_name_list[i]):
					Eng = Eng + ham['w'][i]
		return Eng

	def _evaluate_multi_energy(self, param_list):
		"""
		This is the objective function to be optimized
			returns: Energy, the energy of the system at the given parameters
		"""
		Energy = 0
		for i,eng in enumerate(self.ham_energy_list):
			constant = 1
			if eng != 0:
				if self.dir_to_bracket:
					base_3 = ternary(self.nonzero_terms[i]-1)
				else:
					base_3 = self.base_3_list[i]
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


	def _reconstruct_op_descendants(self, ham_term_list = None):
		"""

		"""
		if ham_term_list == None:
			ham_list= split_into_paulis(self._operator)
			for term in ham_list:
				ham_term_list.append({'term': term, 'descendants': [term], 'tree': []})
		if self.starting_point:
			op_list = self.op_list[-1]
		else:
			op_list = self.op_list
		for num, op in enumerate(op_list): #make sure op list in right direction
			print('num', num)
			if self.starting_point:
				prev_op_list = self.op_list[:-1]
			else:
				prev_op_list = self.op_list[:num]
				
			ham_term_list = parallel_map(sort_term, ham_term_list, num_processes = aqua_globals.num_processes)
		return ham_term_list

	def _generate_base_3_list(self):
		base_3_list = []
		meta_list = parallel_map(generate_base_3_list_single_op, self.ham_term_list, num_processes = aqua_globals.num_processes)
		for term_list in meta_list:
			base_3_list = base_3_list + term_list
		return base_3_list





