from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.algorithms import VQE
import numpy as np
import time
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import SLSQP, NELDER_MEAD
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from CVQE import Classical_VQE as CVQE
from CVQE import find_commutator, split_into_paulis
from qiskit.quantum_info import Pauli
from qisresearch.adapt.adapt_variational_form import ADAPTVariationalForm
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.components.initial_states import InitialState, Zero

import warnings
warnings.filterwarnings("ignore")

def random_pauli(num_qubits):
    """
    creates a random pauli operator
    """
    pauli = ''
    for i in range(0,num_qubits):
        single_pauli_num = np.random.randint(0,4)
        if single_pauli_num == 0:
            pauli = pauli + 'I'
        elif single_pauli_num == 1:
            pauli = pauli + 'X'
        elif single_pauli_num == 2:
            pauli = pauli + 'Y'
        else:
            pauli = pauli + 'Z'
    pauli = Pauli.from_label(pauli)
    return  pauli

def Gen_rand_1_ham(num_terms, num_qubits):
	wop = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
	ham = wop
	for i in range(0,num_terms):
		pauli = random_pauli(num_qubits)
		wpauli = WeightedPauliOperator.from_list(paulis=[pauli], weights=[1.0])
		ham = ham + wpauli
	return ham

def Gen_rand_rand_ham(num_terms, num_qubits):
	wop = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
	ham = wop
	pauli_list = []
	i = 0
	while i < num_terms:
		pauli = random_pauli(num_qubits)
		if pauli_list.count(pauli) == 0:
			wpauli = WeightedPauliOperator.from_list(paulis=[pauli], weights=[np.random.uniform(0,1)])
			ham = ham + wpauli
			pauli_list.append(pauli)
			i = i+1
	return ham