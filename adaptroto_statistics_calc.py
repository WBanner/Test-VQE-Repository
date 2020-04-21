from qisresearch import ADAPTVQE, PauliPool, ADAPTVQEROTO, ROTOEXTENDED, Rotosolve
from qiskit.aqua.components.optimizers import NELDER_MEAD
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.providers.aer import Aer
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator

import numpy as np
import pandas as pd
import math

#def operator_filter(op_pool, ham):
#	for i,op in enumerate(op_pool.pool):
#		if (ham * op + op * ham).chop(threshold=gradient_tolerance, copy=True)  0.0001:
#			print('was deleted')
#			del op_pool.pool[i]
#	return op_pool


out_file = open("ADAPT_ROTO_comparison.txt","w+")

optimizer = NELDER_MEAD(2500)

optimizer_2 = Rotosolve()

backend = Aer.get_backend('statevector_simulator')

#now gotta implement pandas
output_to_file = 0
output_to_cmd = 0
store_in_df = 1
output_to_csv = 1
enable_adapt = 1
enable_roto_1 = 1
enable_roto_2 = 1
enable_roto_extend = 0

number_runs = 20
max_iterations = 10
num_qubits = 2

counter_start = 0
counter = counter_start
ADAPT_ROTO_low_energy_counter = [0, 0, 0, 0]
ADAPT_ROTO_fewer_evals_counter = [0, 0, 0, 0]
ADAPT_ROTO_less_time_counter = [0, 0, 0, 0]
ADAPT_ROTO_fewer_ans_len_counter = [0, 0, 0, 0]

adapt_data_dict = {'hamiltonian': [], 'eval time': [], 'num evals': [], 'ansz length': [], 'final energy': []}
adapt_param_dict = dict()
adapt_op_dict = dict()
adapt_E_dict = dict()

adapt_roto_1_data_dict = {'hamiltonian': [], 'eval time': [], 'num evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_1_param_dict = dict()
adapt_roto_1_op_dict = dict()
adapt_roto_1_E_dict = dict()

adapt_roto_2_data_dict = {'hamiltonian': [], 'eval time': [], 'num evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_2_param_dict = dict()
adapt_roto_2_op_dict = dict()
adapt_roto_2_E_dict = dict()

#we'll iterate through this
while counter <= (number_runs - counter_start - 1):

	shots = 1000
	if output_to_cmd:
		print("shots per measurement: ", shots)
	if output_to_file:
		out_file.write("shots per measurement: {}".format(shots))

	qi = QuantumInstance(backend, shots)

	mat = np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits))
	mat = np.conjugate(np.transpose(mat)) + mat
	ham = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"

	qubit_op = ham

	pool = PauliPool.from_exact_word_length(qubit_op.num_qubits, 2)
	pool = pool + PauliPool.from_exact_word_length(qubit_op.num_qubits, 1)

	if counter == 0:
		if output_to_cmd:
			print("Operator list")
		if output_to_file:
			out_file.write("Operator list:\n")
		for op in pool.pool:
			if output_to_cmd:
				print(op.print_details())
			if output_to_file:
				out_file.write(op.print_details())
	#filt_pool = operator_filter(pool, ham)
	#for op in filt_pool.pool:
	#	print(op.print_details())

	if enable_adapt:
		print('on adapt')
		adapt_vqe = ADAPTVQE(operator_pool=pool, initial_state=None, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations)

		adapt_result = adapt_vqe.run(qi)

		if output_to_cmd:
			print("ADAPT Results for \n{}".format(ham.print_details()))
			print("Total Eval time", adapt_result['Total Eval Time'])
			print("total number of evaluations", adapt_result['Total num evals'])
			print("ansatz length", len(adapt_result['energy_history']))
			print("max gradient list", adapt_result['max_gradient'])
			print("optimal parameters", adapt_result['optimal_parameters'][-1])
			print("operators list", adapt_result['operators'])
			print("energy history", adapt_result['energy_history'])

		#check to make sure adapt is valid
		#for i in range(len(adapt_result['energy_history'])-1):
		#	if adapt_result['energy_history'][i] < (adapt_result['energy_history'][i+1]+0.00000001):
		#		print('we got a problem')
		#		print("energy history", adapt_result['energy_history'])

		if output_to_file:
			out_file.write("ADAPT Results for: \n{}".format(ham.print_details()))
			out_file.write("Total Eval time: {}\n".format(adapt_result['Total Eval Time']))
			out_file.write("total number of evaluations: {}\n".format(adapt_result['Total num evals']))
			out_file.write("ansatz length: {}\n".format(len(adapt_result['energy_history'])))
			out_file.write("max gradient list: {}\n".format(adapt_result['max_gradient']))
			out_file.write("optimal parameters: {}\n".format(adapt_result['optimal_parameters'][-1]))
			out_file.write("operators list: {}\n".format(adapt_result['operators']))
			out_file.write("energy history: {}\n".format(adapt_result['energy_history']))

		if store_in_df:
			adapt_data_dict['hamiltonian'].append(ham.print_details())
			adapt_data_dict['eval time'].append(adapt_result['Total Eval Time'])
			adapt_data_dict['num evals'].append(adapt_result['Total num evals'])
			adapt_data_dict['ansz length'].append(len(adapt_result['energy_history']))
			adapt_data_dict['final energy'].append(adapt_result['energy_history'][-1])

			adapt_param_dict.update({'Ham_{}'.format(counter): adapt_result['optimal_parameters'][-1]})
			adapt_op_dict.update( {'Ham_{}'.format(counter): adapt_result['operators']})
			adapt_E_dict.update({'Ham_{}'.format(counter): adapt_result['energy_history']})


	if enable_roto_1:
		print('on roto 1')
		adapt_vqe_roto = ADAPTVQEROTO(pool, None, optimizer, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False)

		adapt_roto_result = adapt_vqe_roto.run(qi)

		if output_to_cmd:
			print("ADAPT ROTO Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", adapt_roto_result['Total Eval Time'])
			print("total number of evaluations", adapt_roto_result['Total num evals'])
			print("ansatz length", adapt_roto_result['Total number energy iterations'])
			print("max energy step", adapt_roto_result['Max Energy Step'])
			print("optimal parameters", adapt_roto_result['optimal_parameters'])
			print("operator list", adapt_roto_result['operators'])
			print("energy history", adapt_roto_result['energy_history'])

		if output_to_file:
			out_file.write("ADAPT ROTO Results for: \n{}".format(ham.print_details()))
			out_file.write("Total Eval Time: {}\n".format(adapt_roto_result['Total Eval Time']))
			out_file.write("total number of evaluations: {}\n".format(adapt_roto_result['Total num evals']))
			out_file.write("ansatz length {}\n".format(adapt_roto_result['Total number energy iterations']))
			out_file.write("max energy step: {}\n".format(adapt_roto_result['Max Energy Step']))
			out_file.write("optimal parameters: {}\n".format(adapt_roto_result['optimal_parameters']))
			out_file.write("operator list: {}\n".format(adapt_roto_result['operators']))
			out_file.write("energy history: {}\n".format(adapt_roto_result['energy_history']))

		if store_in_df:
			adapt_roto_1_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_1_data_dict['eval time'].append(adapt_roto_result['Total Eval Time'])
			adapt_roto_1_data_dict['num evals'].append(adapt_roto_result['Total num evals'])
			adapt_roto_1_data_dict['ansz length'].append(len(adapt_roto_result['energy_history']))
			adapt_roto_1_data_dict['final energy'].append(adapt_roto_result['energy_history'][-1])

			adapt_roto_1_param_dict.update({'Ham_{}'.format(counter): adapt_roto_result['optimal_parameters'][-1]})
			adapt_roto_1_op_dict.update( {'Ham_{}'.format(counter): adapt_roto_result['operators']})
			adapt_roto_1_E_dict.update({'Ham_{}'.format(counter): adapt_roto_result['energy_history']})

	if enable_roto_2:
		print('on roto 1')
		adapt_roto_2 = ADAPTVQEROTO(pool, None, optimizer_2, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False, postprocessing = True)

		adapt_roto_2_result = adapt_roto_2.run(qi)

		if output_to_cmd:
			print("ADAPT ROTO 2 Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", adapt_roto_2_result['Total Eval Time'])
			print("total number of evaluations", adapt_roto_2_result['Total num evals'])
			print("ansatz length", adapt_roto_2_result['Total number energy iterations'])
			print("max energy step", adapt_roto_2_result['Max Energy Step'])
			print("optimal parameters", adapt_roto_2_result['optimal_parameters'][-1])
			print("operator list", adapt_roto_2_result['operators'])
			print("energy history", adapt_roto_2_result['energy_history'])

		if output_to_file:
			out_file.write("ADAPT ROTO Results for: \n{}".format(ham.print_details()))
			out_file.write("Total Eval Time: {}\n".format(adapt_roto_2_result['Total Eval Time']))
			out_file.write("total number of evaluations: {}\n".format(adapt_roto_2_result['Total num evals']))
			out_file.write("ansatz length {}\n".format(adapt_roto_2_result['Total number energy iterations']))
			out_file.write("max energy step: {}\n".format(adapt_roto_2_result['Max Energy Step']))
			out_file.write("optimal parameters: {}\n".format(adapt_roto_2_result['optimal_parameters'][-1]))
			out_file.write("operator list: {}\n".format(adapt_roto_2_result['operators']))
			out_file.write("energy history: {}\n".format(adapt_roto_2_result['energy_history']))

		if store_in_df:
			adapt_roto_2_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_2_data_dict['eval time'].append(adapt_roto_2_result['Total Eval Time'])
			adapt_roto_2_data_dict['num evals'].append(adapt_roto_2_result['Total num evals'])
			adapt_roto_2_data_dict['ansz length'].append(len(adapt_roto_2_result['energy_history']))
			adapt_roto_2_data_dict['final energy'].append(adapt_roto_2_result['energy_history'][-1])

			adapt_roto_2_param_dict.update({'Ham_{}'.format(counter): adapt_roto_2_result['optimal_parameters'][-1]})
			adapt_roto_2_op_dict.update( {'Ham_{}'.format(counter): adapt_roto_2_result['operators']})
			adapt_roto_2_E_dict.update({'Ham_{}'.format(counter): adapt_roto_2_result['energy_history']})

	if enable_roto_extend:

		adapt_roto_extend = ROTOEXTENDED(pool, None, optimizer, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False)

		adapt_roto_extend_result = adapt_roto_extend.run(qi)

		if output_to_cmd:
			print("ADAPT ROTO Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", adapt_roto_extend_result['Total Eval Time'])
			print("total number of evaluations", adapt_roto_extend_result['Total num evals'])
			print("ansatz length", adapt_roto_extend_result['Total number energy iterations'])
			print("max energy step", adapt_roto_extend_result['Max Energy Step'])
			print("optimal parameters", adapt_roto_extend_result['optimal_parameters'])
			print("operator list", adapt_roto_extend_result['operators'])
			print("energy history", adapt_roto_extend_result['energy_history'])
			print("A", adapt_roto_extend_result['A'])
			print("B", adapt_roto_extend_result['B'])
			print("C", adapt_roto_extend_result['C'])

	#metadata stuff
	#if enable_roto_1 and enable_adapt:
	#	if adapt_roto_result['energy_history'][-1] < adapt_result['energy_history'][-1]:
	#		ADAPT_ROTO_low_energy_counter[math.floor(counter/10)] += 1
	#		ADAPT_ROTO_low_energy_counter[3] += 1
#
#		if adapt_roto_result['Total num evals'] < adapt_result['Total num evals']:
#			ADAPT_ROTO_fewer_evals_counter[math.floor(counter/10)] +=1
#			ADAPT_ROTO_fewer_evals_counter[3] +=1
#
#		if adapt_roto_result['Total Eval Time'] < adapt_result['Total Eval Time']:
#			ADAPT_ROTO_less_time_counter[math.floor(counter/10)] += 1
#			ADAPT_ROTO_less_time_counter[3] += 1
#
#		if adapt_roto_result['Total number energy iterations'] < len(adapt_result['energy_history']):
#			ADAPT_ROTO_fewer_ans_len_counter[(counter%10)+1] += 1
#			ADAPT_ROTO_fewer_ans_len_counter[3] += 1

	counter += 1
	print("counter", counter)

#if output_to_file:
#	out_file.write("ADAPTROTO less energy count: {}".format(ADAPT_ROTO_low_energy_counter))
#	out_file.write("ADAPTROTO less eval num count: {}".format(ADAPT_ROTO_fewer_evals_counter))
#	out_file.write("ADAPTROTO less time count: {}".format(ADAPT_ROTO_less_time_counter))
#	out_file.write("ADAPTROTO smaller ansatz count: {}".format(ADAPT_ROTO_fewer_ans_len_counter))

adapt_data_df = pd.DataFrame(adapt_data_dict)
adapt_param_df = pd.DataFrame(adapt_param_dict)
adapt_op_df = pd.DataFrame(adapt_op_dict)
adapt_E_df = pd.DataFrame(adapt_E_dict)

adapt_roto_1_data_df = pd.DataFrame(adapt_roto_1_data_dict)
adapt_roto_1_param_df = pd.DataFrame(adapt_roto_1_param_dict)
adapt_roto_1_op_df = pd.DataFrame(adapt_roto_1_op_dict)
adapt_roto_1_E_df = pd.DataFrame(adapt_roto_1_E_dict)

adapt_roto_2_data_df = pd.DataFrame(adapt_roto_2_data_dict)
adapt_roto_2_param_df = pd.DataFrame(adapt_roto_2_param_dict)
adapt_roto_2_op_df = pd.DataFrame(adapt_roto_2_op_dict)
adapt_roto_2_E_df = pd.DataFrame(adapt_roto_2_E_dict)


if output_to_csv:
	adapt_data_df_file = open("adapt_data_df.csv","w+")
	adapt_param_df_file = open("adapt_param_df.csv","w+")
	adapt_op_df_file = open("adapt_op_df.csv","w+")
	adapt_E_df_file = open("adapt_E_df.csv","w+")

	adapt_roto_1_data_df_file = open("adapt_roto_1_data_df.csv","w+")
	adapt_roto_1_param_df_file = open("adapt_roto_1_param_df.csv","w+")
	adapt_roto_1_op_df_file = open("adapt_roto_1_op_df.csv","w+")
	adapt_roto_1_E_df_file = open("adapt_roto_1_E_df.csv","w+")

	adapt_roto_2_data_df_file = open("adapt_roto_2_data_df.csv","w+")
	adapt_roto_2_param_df_file = open("adapt_roto_2_param_df.csv","w+")
	adapt_roto_2_op_df_file = open("adapt_roto_2_op_df.csv","w+")
	adapt_roto_2_E_df_file = open("adapt_roto_2_E_df.csv","w+")

	adapt_data_df.to_csv('adapt_data_df.csv')
	adapt_param_df.to_csv('adapt_param_df.csv')
	adapt_op_df.to_csv('adapt_op_df.csv')
	adapt_E_df.to_csv('adapt_E_df_file.csv')

	adapt_roto_1_data_df.to_csv('adapt_roto_1_data_df.csv')
	adapt_roto_1_param_df.to_csv('adapt_roto_1_param_df.csv')
	adapt_roto_1_op_df.to_csv('adapt_roto_1_op_df.csv')
	adapt_roto_1_E_df.to_csv('adapt_roto_1_E_df.csv')

	adapt_roto_2_data_df.to_csv('adapt_roto_2_data_df.csv')
	adapt_roto_2_param_df.to_csv('adapt_roto_2_param_df.csv')
	adapt_roto_2_op_df.to_csv('adapt_roto_2_op_df.csv')
	adapt_roto_2_E_df.to_csv('adapt_roto_2_E_df.csv')

	adapt_data_df_file.close()
	adapt_param_df_file.close()
	adapt_op_df_file.close()
	adapt_E_df_file.close()

	adapt_roto_1_data_df_file.close()
	adapt_roto_1_param_df_file.close()
	adapt_roto_1_op_df_file.close()
	adapt_roto_1_E_df_file.close()

	adapt_roto_2_data_df_file.close()
	adapt_roto_2_param_df_file.close()
	adapt_roto_2_op_df_file.close()
	adapt_roto_2_E_df_file.close()

out_file.close()










