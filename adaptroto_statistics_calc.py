from qisresearch.adapt import ADAPTVQE, PauliPool, ADAPTVQEROTO

from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.providers.aer import Aer
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator

import numpy as np
import math

out_file = open("ADAPT_ROTO_comparison.txt","w+")

num_qubits = 2

optimizer = COBYLA(maxiter=100)

max_iterations = 9

backend = Aer.get_backend('statevector_simulator')

ADAPT_ROTO_low_energy_counter = [0, 0, 0, 0]
ADAPT_ROTO_fewer_evals_counter = [0, 0, 0, 0]
ADAPT_ROTO_less_time_counter = [0, 0, 0, 0]
ADAPT_ROTO_fewer_ans_len_counter = [0, 0, 0, 0]

adapt_error_flag = 0
counter = 0


#we'll iterate through this
while counter <= 29:
	if counter < 10:
		shots = 750
	elif counter >= 10 and counter < 20:
		shots = 1500
	else:
		shots = 3000
	#print("shots per measurement: ", shots)
	out_file.write("shots per measurement: {}".format(shots))

	qi = QuantumInstance(backend, shots)

	mat = np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits))
	mat = np.conjugate(np.transpose(mat)) + mat
	ham = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"

	qubit_op = ham

	pool = PauliPool.from_exact_word_length(qubit_op.num_qubits, 2)
	pool = pool + PauliPool.from_exact_word_length(qubit_op.num_qubits, 1)

	if counter == 0:
		#print("Operator list")
		out_file.write("Operator list:\n")
		for op in pool.pool:
		   #print(op.print_details())
		   out_file.write(op.print_details())

	adapt_vqe = ADAPTVQE(operator_pool=pool, initial_state=None, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations)

	adapt_result = adapt_vqe.run(qi)

	#print("ADAPT Results for \n{}".format(ham.print_details()))
	#print("Total Eval time", adapt_result['Total Eval Time'])
	#print("total number of evaluations", adapt_result['Total num evals'])
	#print("ansatz length", len(adapt_result['energy_history']))
	#print("max gradient list", adapt_result['max_gradient'])
	#print("optimal parameters", adapt_result['optimal_parameters'][-1])
	#print("operators list", adapt_result['operators'])
	#print("energy history", adapt_result['energy_history'])

	#check to make sure adapt is valid
	for i in range(len(adapt_result['energy_history'])-1):
		if adapt_result['energy_history'][i] < (adapt_result['energy_history'][i+1]+0.00000001):
			adapt_error_flag = 1

	if adapt_error_flag == 0:
		out_file.write("ADAPT Results for: \n{}".format(ham.print_details()))
		out_file.write("Total Eval time: {}\n".format(adapt_result['Total Eval Time']))
		out_file.write("total number of evaluations: {}\n".format(adapt_result['Total num evals']))
		out_file.write("ansatz length: {}\n".format(len(adapt_result['energy_history'])))
		out_file.write("max gradient list: {}\n".format(adapt_result['max_gradient']))
		out_file.write("optimal parameters: {}\n".format(adapt_result['optimal_parameters'][-1]))
		out_file.write("operators list: {}\n".format(adapt_result['operators']))
		out_file.write("energy history: {}\n".format(adapt_result['energy_history']))

		adapt_vqe_roto = ADAPTVQEROTO(pool, None, optimizer, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False)

		adapt_roto_result = adapt_vqe_roto.run(qi)

		#print("ADAPT ROTO Results for \n{}".format(ham.print_details()))
		#print("Total Eval Time", adapt_roto_result['Total Eval Time'])
		#print("total number of evaluations", adapt_roto_result['Total num evals'])
		#print("ansatz length", adapt_roto_result['Total number energy iterations'])
		#print("max energy step", adapt_roto_result['Max Energy Step'])
		#print("optimal parameters", adapt_roto_result['optimal_parameters'])
		#print("operator list", adapt_roto_result['operators'])
		#print("energy history", adapt_roto_result['energy_history'])

		out_file.write("ADAPT ROTO Results for: \n{}".format(ham.print_details()))
		out_file.write("Total Eval Time: {}\n".format(adapt_roto_result['Total Eval Time']))
		out_file.write("total number of evaluations: {}\n".format(adapt_roto_result['Total num evals']))
		out_file.write("ansatz length {}\n".format(adapt_roto_result['Total number energy iterations']))
		out_file.write("max energy step: {}\n".format(adapt_roto_result['Max Energy Step']))
		out_file.write("optimal parameters: {}\n".format(adapt_roto_result['optimal_parameters']))
		out_file.write("operator list: {}\n".format(adapt_roto_result['operators']))
		out_file.write("energy history: {}\n".format(adapt_roto_result['energy_history']))

		#metadata stuff

		if adapt_roto_result['energy_history'][-1] < adapt_result['energy_history'][-1]:
			ADAPT_ROTO_low_energy_counter[math.floor(counter/10)] += 1
			ADAPT_ROTO_low_energy_counter[3] += 1

		if adapt_roto_result['Total num evals'] < adapt_result['Total num evals']:
			ADAPT_ROTO_fewer_evals_counter[math.floor(counter/10)] +=1
			ADAPT_ROTO_fewer_evals_counter[3] +=1

		if adapt_roto_result['Total Eval Time'] < adapt_result['Total Eval Time']:
			ADAPT_ROTO_less_time_counter[math.floor(counter/10)] += 1
			ADAPT_ROTO_less_time_counter[3] += 1

		if adapt_roto_result['Total number energy iterations'] < len(adapt_result['energy_history']):
			ADAPT_ROTO_fewer_ans_len_counter[(counter%10)+1] += 1
			ADAPT_ROTO_fewer_ans_len_counter[3] += 1

		counter += 1

	else:
		adapt_error_flag = 0

	print("counter", counter)

out_file.write("ADAPTROTO less energy count: {}".format(ADAPT_ROTO_low_energy_counter))
out_file.write("ADAPTROTO less eval num count: {}".format(ADAPT_ROTO_fewer_evals_counter))
out_file.write("ADAPTROTO less time count: {}".format(ADAPT_ROTO_less_time_counter))
out_file.write("ADAPTROTO smaller ansatz count: {}".format(ADAPT_ROTO_fewer_ans_len_counter))
out_file.close()






