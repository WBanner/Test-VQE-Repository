from qisresearch import ADAPTVQE, PauliPool, ADAPTVQEROTO, ROTOEXTENDED, Rotosolve
from qiskit.aqua.components.optimizers import NELDER_MEAD
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.providers.aer import Aer
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
from qiskit import IBMQ

import numpy as np
import pandas as pd
import math
import datetime

starttime = datetime.datetime.now()


backend = Aer.get_backend('statevector_simulator')

shots = 1000 #doesn't matter for statevector simulator 

qi = QuantumInstance(backend, shots)

output_to_file = 1
output_to_cmd = 0 
store_in_df = 1
output_to_csv = 1
enable_adapt = 1
enable_roto_1 = 1
enable_roto_2 = 1
enable_roto_3 = 1
enable_roto_extend = 0
num_optimizer_runs = 100000


print(starttime)
number_runs = 1
max_iterations = 6
ADAPT_stopping_gradient = 1e-3 #not used
ADAPTROTO_stopping_energy = 1e-8 #not used
ROTOSOLVE_stopping_energy = 1e-4
ADAPT_optimizer_stopping_energy = 1e-4
ROTOSOLVE_max_iterations = 100000

num_qubits = 4

out_file = open("ADAPT_ROTO_RUN_INFO.txt","w+")

optimizer_name = "NELDER_MEAD"
optimizer = NELDER_MEAD(tol = ADAPT_optimizer_stopping_energy)

optimizer_2 = Rotosolve(ROTOSOLVE_stopping_energy,ROTOSOLVE_max_iterations)

counter_start = 0
counter = counter_start

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

adapt_roto_3_data_dict = {'hamiltonian': [], 'eval time': [], 'num evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_3_param_dict = dict()
adapt_roto_3_op_dict = dict()
adapt_roto_3_E_dict = dict()

#we'll iterate through this


while counter <= (number_runs - counter_start - 1):

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

	if enable_adapt:
		print('on adapt')
		adapt_vqe = ADAPTVQE(operator_pool=pool, initial_state=None, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations, grad_tol = ADAPT_stopping_gradient)

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
		adapt_vqe_roto = ADAPTVQEROTO(pool, None, optimizer, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False, energy_step_tol = ADAPTROTO_stopping_energy)

		adapt_roto_result = adapt_vqe_roto.run(qi)

		if output_to_cmd:
			print("ADAPT ROTO Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", adapt_roto_result['Total Eval Time'])
			print("total number of evaluations", adapt_roto_result['Total num evals'])
			print("ansatz length", adapt_roto_result['Total number energy iterations'])
			print("max energy step", adapt_roto_result['Max Energy Step'])
			print("optimal parameters", adapt_roto_result['optimal_parameters'][-1])
			print("operator list", adapt_roto_result['operators'])
			print("energy history", adapt_roto_result['energy_history'])

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
		print('on roto 2')
		adapt_roto_2 = ADAPTVQEROTO(pool, None, optimizer_2, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False, energy_step_tol = ADAPTROTO_stopping_energy, postprocessing = True)

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

		if store_in_df:
			adapt_roto_2_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_2_data_dict['eval time'].append(adapt_roto_2_result['Total Eval Time'])
			adapt_roto_2_data_dict['num evals'].append(adapt_roto_2_result['Total num evals'])
			adapt_roto_2_data_dict['ansz length'].append(len(adapt_roto_2_result['energy_history']))
			adapt_roto_2_data_dict['final energy'].append(adapt_roto_2_result['energy_history'][-1])

			adapt_roto_2_param_dict.update({'Ham_{}'.format(counter): adapt_roto_2_result['optimal_parameters'][-1]})
			adapt_roto_2_op_dict.update( {'Ham_{}'.format(counter): adapt_roto_2_result['operators']})
			adapt_roto_2_E_dict.update({'Ham_{}'.format(counter): adapt_roto_2_result['energy_history']})

	if enable_roto_3:
		print('on roto 3')
		adapt_roto_3 = ADAPTVQEROTO(pool, None, optimizer, qubit_op, max_iterations, auto_conversion=True, use_zero_initial_parameters=False, energy_step_tol = ADAPTROTO_stopping_energy, postprocessing = True, include_parameter = False)

		adapt_roto_3_result = adapt_roto_3.run(qi)

		if output_to_cmd:
			print("ADAPT ROTO 3 Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", adapt_roto_3_result['Total Eval Time'])
			print("total number of evaluations", adapt_roto_3_result['Total num evals'])
			print("ansatz length", adapt_roto_3_result['Total number energy iterations'])
			print("max energy step", adapt_roto_3_result['Max Energy Step'])
			print("optimal parameters", adapt_roto_3_result['optimal_parameters'][-1])
			print("operator list", adapt_roto_3_result['operators'])
			print("energy history", adapt_roto_3_result['energy_history'])

		if store_in_df:
			adapt_roto_3_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_3_data_dict['eval time'].append(adapt_roto_3_result['Total Eval Time'])
			adapt_roto_3_data_dict['num evals'].append(adapt_roto_3_result['Total num evals'])
			adapt_roto_3_data_dict['ansz length'].append(len(adapt_roto_3_result['energy_history']))
			adapt_roto_3_data_dict['final energy'].append(adapt_roto_3_result['energy_history'][-1])

			adapt_roto_3_param_dict.update({'Ham_{}'.format(counter): adapt_roto_3_result['optimal_parameters'][-1]})
			adapt_roto_3_op_dict.update( {'Ham_{}'.format(counter): adapt_roto_3_result['operators']})
			adapt_roto_3_E_dict.update({'Ham_{}'.format(counter): adapt_roto_3_result['energy_history']})


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

	counter += 1
	print("counter", counter)

if output_to_csv:
	if enable_adapt:
		adapt_data_df = pd.DataFrame(adapt_data_dict)
		adapt_param_df = pd.DataFrame(adapt_param_dict)
		adapt_op_df = pd.DataFrame(adapt_op_dict)
		adapt_E_df = pd.DataFrame(adapt_E_dict)

		adapt_data_df_file = open("adapt_data_df.csv","w+")
		adapt_param_df_file = open("adapt_param_df.csv","w+")
		adapt_op_df_file = open("adapt_op_df.csv","w+")
		adapt_E_df_file = open("adapt_E_df.csv","w+")

		adapt_data_df.to_csv('adapt_data_df.csv')
		adapt_param_df.to_csv('adapt_param_df.csv')
		adapt_op_df.to_csv('adapt_op_df.csv')
		adapt_E_df.to_csv('adapt_E_df_file.csv')

		adapt_data_df_file.close()
		adapt_param_df_file.close()
		adapt_op_df_file.close()
		adapt_E_df_file.close()

	if enable_roto_1:
		adapt_roto_1_data_df = pd.DataFrame(adapt_roto_1_data_dict)
		adapt_roto_1_param_df = pd.DataFrame(adapt_roto_1_param_dict)
		adapt_roto_1_op_df = pd.DataFrame(adapt_roto_1_op_dict)
		adapt_roto_1_E_df = pd.DataFrame(adapt_roto_1_E_dict)

		adapt_roto_1_data_df_file = open("adapt_roto_1_data_df.csv","w+")
		adapt_roto_1_param_df_file = open("adapt_roto_1_param_df.csv","w+")
		adapt_roto_1_op_df_file = open("adapt_roto_1_op_df.csv","w+")
		adapt_roto_1_E_df_file = open("adapt_roto_1_E_df.csv","w+")

		adapt_roto_1_data_df.to_csv('adapt_roto_1_data_df.csv')
		adapt_roto_1_param_df.to_csv('adapt_roto_1_param_df.csv')
		adapt_roto_1_op_df.to_csv('adapt_roto_1_op_df.csv')
		adapt_roto_1_E_df.to_csv('adapt_roto_1_E_df.csv')

		adapt_roto_1_data_df_file.close()
		adapt_roto_1_param_df_file.close()
		adapt_roto_1_op_df_file.close()
		adapt_roto_1_E_df_file.close()

	if enable_roto_2:
		adapt_roto_2_data_df = pd.DataFrame(adapt_roto_2_data_dict)
		adapt_roto_2_param_df = pd.DataFrame(adapt_roto_2_param_dict)
		adapt_roto_2_op_df = pd.DataFrame(adapt_roto_2_op_dict)
		adapt_roto_2_E_df = pd.DataFrame(adapt_roto_2_E_dict)

		adapt_roto_2_data_df_file = open("adapt_roto_2_data_df.csv","w+")
		adapt_roto_2_param_df_file = open("adapt_roto_2_param_df.csv","w+")
		adapt_roto_2_op_df_file = open("adapt_roto_2_op_df.csv","w+")
		adapt_roto_2_E_df_file = open("adapt_roto_2_E_df.csv","w+")

		adapt_roto_2_data_df.to_csv('adapt_roto_2_data_df.csv')
		adapt_roto_2_param_df.to_csv('adapt_roto_2_param_df.csv')
		adapt_roto_2_op_df.to_csv('adapt_roto_2_op_df.csv')
		adapt_roto_2_E_df.to_csv('adapt_roto_2_E_df.csv')

		adapt_roto_2_data_df_file.close()
		adapt_roto_2_param_df_file.close()
		adapt_roto_2_op_df_file.close()
		adapt_roto_2_E_df_file.close()

	if enable_roto_3:
		adapt_roto_3_data_df = pd.DataFrame(adapt_roto_3_data_dict)
		adapt_roto_3_param_df = pd.DataFrame(adapt_roto_3_param_dict)
		adapt_roto_3_op_df = pd.DataFrame(adapt_roto_3_op_dict)
		adapt_roto_3_E_df = pd.DataFrame(adapt_roto_3_E_dict)

		adapt_roto_3_data_df_file = open("adapt_roto_3_data_df.csv","w+")
		adapt_roto_3_param_df_file = open("adapt_roto_3_param_df.csv","w+")
		adapt_roto_3_op_df_file = open("adapt_roto_3_op_df.csv","w+")
		adapt_roto_3_E_df_file = open("adapt_roto_3_E_df.csv","w+")

		adapt_roto_3_data_df.to_csv('adapt_roto_3_data_df.csv')
		adapt_roto_3_param_df.to_csv('adapt_roto_3_param_df.csv')
		adapt_roto_3_op_df.to_csv('adapt_roto_3_op_df.csv')
		adapt_roto_3_E_df.to_csv('adapt_roto_3_E_df.csv')

		adapt_roto_3_data_df_file.close()
		adapt_roto_3_param_df_file.close()
		adapt_roto_3_op_df_file.close()
		adapt_roto_3_E_df_file.close()

stoptime = datetime.datetime.now()

if output_to_file:
	out_file.write("Analysis start time: {}\n".format(starttime))
	out_file.write("Analysis stop time: {}\n".format(stoptime))
	out_file.write("Number of qubits: {}\n".format(num_qubits))
	out_file.write("Number of runs in this set: {}\n".format(number_runs))
	out_file.write("Max number of energy steps: {}\n".format(max_iterations))
	out_file.write("Shots per measurement: {}\n".format(shots))
	if enable_adapt:
			out_file.write("ADAPT enabled\n")
			out_file.write("Optimizer: {}\n".format(optimizer_name))
			out_file.write("Max optimzer iterations: {}\n".format(num_optimizer_runs))
			out_file.write("ADAPT stopping gradient < {}\n".format(ADAPT_stopping_gradient))
	if enable_roto_1:
			out_file.write("ADAPTROTO enabled\n")
			out_file.write("ADAPTROTO stopping energy change < {}\n".format(ADAPTROTO_stopping_energy))
	if enable_roto_2:
			out_file.write("ADAPTROTO with postprocessing enabled\n")
			out_file.write("ADAPT_ROTO with postproceesing stopping energy change < {}\n".format(ADAPTROTO_stopping_energy))
			out_file.write("Rotosolve optimizer max iterations: {}".format(ROTOSOLVE_max_iterations))
	if enable_roto_3:
			out_file.write("ADAPTROTO with postprocessing enabled and different optimizer\n")
			out_file.write("Optimizer: {}\n".format(optimizer_name))
			out_file.write("ADAPT_ROTO with postproceesing stopping energy change < {}\n".format(ADAPTROTO_stopping_energy))
			out_file.write("optimizer max iterations: {}".format(num_optimizer_runs))
	if enable_roto_extend:
			out_file.write("2 parameter ADAPTROTO enabled\n")


out_file.close()










