import logging
import functools
from copy import deepcopy, copy
import random

import pandas as pd
from qiskit import QuantumRegister
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool
from qiskit.providers.aer import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator, WeightedPauliOperator,
                                   MatrixOperator, op_converter)
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_provider)
from qiskit import IBMQ
from qiskit.aqua.components.initial_states import InitialState, Zero, Custom
from operator_selector_new import multi_circuit_eval

import warnings
warnings.simplefilter("ignore")




backend = Aer.get_backend('statevector_simulator')
qi = QuantumInstance(backend)

pool_info = {'names': [], 'exp vals': []}
evals = 0

num_qubits = 4

pool = PauliPool.from_all_pauli_strings(num_qubits) #all possible pauli strings
q = QuantumRegister(num_qubits, name='q')

circ = Zero(num_qubits).construct_circuit('circuit', q)

for op in pool.pool:
	pool_info['names'].append(op.print_details())

pool_info['exp vals'], evals = multi_circuit_eval(circ, pool.pool, qi, True)

for i in range(0, (len(pool_info['exp vals']))):
	pool_info['exp vals'][i] = complex(pool_info['exp vals'][i][0])
	
print('evals', evals)

out_file = open("Pauli_values.csv","w+")

pool_info_df = pd.DataFrame(pool_info)
pool_info_df.to_csv('Pauli_values.csv')
out_file.close()

