from ROTOADAPT_file_pauli import ROTOADAPTVQE
from adapt_new import ADAPTVQE
#from rotosolve_edited import Rotosolve
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool
from qiskit.aqua.components.optimizers import NELDER_MEAD, L_BFGS_B, COBYLA
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.quantum_info import Pauli
from qiskit.providers.aer import Aer
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
from qisresearch.utils.compare_results import add_exact_comparison
from qisresearch.utils.random_hamiltonians import random_diagonal_hermitian, get_h_4_hamiltonian
from mol_ham_file import get_qubit_op
from qiskit import IBMQ
import psutil
import sys
import numpy as np
import pandas as pd
import scipy
import math
import datetime
import time
from qiskit.aqua import aqua_globals, QuantumInstance
import warnings
from qiskit.tools import parallel_map

up = sys.argv[1]
warnings.simplefilter("ignore")
starttime = datetime.datetime.now()
print(starttime)


backend = Aer.get_backend('statevector_simulator')

shots = 1 #doesn't matter for statevector simulator 

qi = QuantumInstance(backend, shots)

output_to_file = 1
output_to_cmd = 1
store_in_df = 1
output_to_csv = 1
enable_adapt = 1
enable_roto_2 = 1

max_iterations = 1


num_optimizer_runs = 100000
optimizer_stopping_energy = 1e-16
optimizer_name = "NM"
optimizer = NELDER_MEAD(tol = optimizer_stopping_energy)

out_file = open("ADAPT_ROTO_RUN_INFO_{}.txt".format(up),"w+")

adapt_data_dict = {'hamiltonian': [], 'eval time': [], 'num op choice evals': [], 'num optimizer evals': [], 'ansz length': [], 'final energy': []}
adapt_param_dict = dict()
adapt_op_dict = dict()
adapt_E_dict = dict()
adapt_grad_dict = dict()
Exact_energy_dict = {'ground energy':[]}


adapt_roto_2_data_dict = {'hamiltonian': [], 'eval time': [], 'num optimizer evals': [], 'num op choice evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_2_param_dict = dict()
adapt_roto_2_op_dict = dict()
adapt_roto_2_E_dict = dict()
adapt_roto_2_counter_dict = dict()
Exact_energy_dict = {'ground energy':[]}


dist = [1.0, 1.5]

number_runs = len(dist)


def generate_ham_pool(dist):
	ham, num_particles, num_spin_orbitals, shift = get_qubit_op(dist)
	return ham

ham_list = list(parallel_map(generate_ham_pool, dist, num_processes = aqua_globals.num_processes))
num_qubits = ham_list[0].num_qubits
print('num qubits', num_qubits)
start = time.time()
pool = PauliPool.from_all_pauli_strings(num_qubits) #all possible pauli strings
pool.cast_out_even_y_paulis(True) #more efficient pool
#pool.cast_out_higher_order_z_strings(True)
#pool.cast_out_particle_number_violating_strings(True)
gentime = time.time() - start
print('done generating pool', gentime)

def generate_exact_result(ham):
	qubit_op = ham
	Exact_result = ExactEigensolver(qubit_op).run()
	return Exact_result

Exact_energy_dict['ground energy'] = list(parallel_map(generate_exact_result, ham_list, num_processes = aqua_globals.num_processes))



if output_to_cmd:
	for i in range(0,len(dist)):
		print("Exact Energy", Exact_energy_dict['ground energy'][i]['energy'])
if output_to_file:
	for i in range(0,len(dist)):
		out_file.write("Exact Energy: {}\n".format(Exact_energy_dict['ground energy'][i]['energy']))


def run_multi_adapt(ham, **kwargs):
	shots = 1 #doesn't matter for statevector simulator 
	qi = QuantumInstance(kwargs['backend'], shots)
	print('on adapt')
	adapt_vqe = ADAPTVQE(operator_pool=kwargs['pool'], initial_state=None, vqe_optimizer=kwargs['optimizer'], hamiltonian=ham, max_iters = kwargs['max_iterations'], grad_tol = 0)
	start = time.time()
	adapt_result = adapt_vqe.run(qi)
	eval_time = time.time() - start

	return [adapt_result, eval_time]

	
def run_multi_roto(ham, **kwargs):
	shots = 1 #doesn't matter for statevector simulator 
	qi = QuantumInstance(kwargs['backend'], shots)
	print('on roto 2')
	adapt_roto_2 = ROTOADAPTVQE(operator_pool=kwargs['pool'], initial_state=None, vqe_optimizer=kwargs['optimizer'], hamiltonian=ham, max_iters = kwargs['max_iterations'], initial_parameters = 0)
	start = time.time()
	adapt_roto_2_result = adapt_roto_2.run(qi)
	eval_time = time.time() - start

	return [adapt_roto_2_result, eval_time]


kwargs = {'output_to_cmd': output_to_cmd, 'output_to_file': output_to_file, 'optimizer': optimizer, 'max_iterations': max_iterations, 'pool': pool, 'backend': backend}
adapt_results = list(parallel_map(run_multi_adapt, ham_list, task_kwargs = kwargs, num_processes = aqua_globals.num_processes))
roto_results = list(parallel_map(run_multi_roto, ham_list, task_kwargs = kwargs, num_processes = aqua_globals.num_processes))

if enable_adapt:
	for counter,result in enumerate(adapt_results):
		adapt_result = result[0]
		eval_time = result[1]
		ham = ham_list[counter]
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		grad_list = []
		for i in range(0,(len(adapt_result))):
			num_op_evals = num_op_evals + adapt_result[i]['eval_count']
			num_op_choice_evals = num_op_choice_evals + adapt_result[i]['num op choice evals']
			energy_history.append(adapt_result[i]['energy'])
			op_list.append(adapt_result[-1]['current_ops'][i].print_details())
			grad_list.append(adapt_result[i]['max_grad'])

		if output_to_cmd:
			print("ADAPT Results for \n{}".format(ham.print_details()))
			print("Total Eval time", eval_time)
			print("total number of op evaluations", num_op_evals)
			print("total number of op choice evals", num_op_choice_evals)
			print("ansatz length", len(adapt_result))
			print("max gradient list", grad_list)
			print("optimal parameters", adapt_result[-1]['opt_params'])
			print("operators list", op_list)
			print("energy history", energy_history)

		if store_in_df:
			adapt_data_dict['hamiltonian'].append(ham.print_details())
			adapt_data_dict['eval time'].append(eval_time)
			adapt_data_dict['num optimizer evals'].append(num_op_evals)
			adapt_data_dict['num op choice evals'].append(num_op_choice_evals)
			adapt_data_dict['ansz length'].append(len(adapt_result))
			adapt_data_dict['final energy'].append(energy_history[-1])
			adapt_param_dict.update({'Ham_{}'.format(counter): adapt_result[-1]['opt_params']})
			adapt_op_dict.update( {'Ham_{}'.format(counter): op_list})
			adapt_E_dict.update({'Ham_{}'.format(counter): energy_history})
			adapt_grad_dict.update({'Ham_{}'.format(counter): grad_list})

if enable_roto_2:
	for counter,result in enumerate(roto_results):
		adapt_roto_2_result = result[0]
		eval_time = result[1]
		ham = ham_list[counter]
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		counter_list = []
		for i in range(0,(len(adapt_roto_2_result))):
			num_op_evals = num_op_evals + adapt_roto_2_result[i]['eval_count']
			num_op_choice_evals = num_op_choice_evals + adapt_roto_2_result[i]['num op choice evals']
			energy_history.append(adapt_roto_2_result[i]['energy'])
			op_list.append(adapt_roto_2_result[-1]['current_ops'][i].print_details())
			counter_list.append(adapt_roto_2_result[i]['optimizer_counter'])
		print(energy_history)


		if output_to_cmd:
			print("ADAPT ROTO 2 Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", eval_time)
			print("total number of op evaluations", num_op_evals)
			print("total number of op choice evals", num_op_choice_evals)
			print("ansatz length", len(adapt_roto_2_result))
			print("optimal parameters", adapt_roto_2_result[-1]['opt_params'])
			print("operator list", op_list)
			print("energy history", energy_history)

		if store_in_df:
			adapt_roto_2_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_2_data_dict['eval time'].append(eval_time)
			adapt_roto_2_data_dict['num optimizer evals'].append(num_op_evals)
			adapt_roto_2_data_dict['num op choice evals'].append(num_op_choice_evals)
			adapt_roto_2_data_dict['ansz length'].append(len(adapt_roto_2_result))
			adapt_roto_2_data_dict['final energy'].append(energy_history[-1])

			adapt_roto_2_param_dict.update({'Ham_{}'.format(counter): adapt_roto_2_result[-1]['opt_params']})
			adapt_roto_2_op_dict.update( {'Ham_{}'.format(counter): op_list})
			adapt_roto_2_E_dict.update({'Ham_{}'.format(counter): energy_history})
			adapt_roto_2_counter_dict.update({'Ham_{}'.format(counter): counter_list})


if output_to_csv:
	Exact_energy_df = pd.DataFrame(Exact_energy_dict)
	Exact_energy_df_file = open("exact_energy_df.csv".format(up), "w+")
	Exact_energy_df.to_csv('exact_energy_df.csv'.format(up))
	Exact_energy_df_file.close()
	if enable_adapt:
		adapt_data_df = pd.DataFrame(adapt_data_dict)
		adapt_param_df = pd.DataFrame(adapt_param_dict)
		adapt_op_df = pd.DataFrame(adapt_op_dict)
		adapt_E_df = pd.DataFrame(adapt_E_dict)
		adapt_grad_df = pd.DataFrame(adapt_grad_dict)

		adapt_data_df.to_csv('adapt_data_df.csv')
		adapt_param_df.to_csv('adapt_param_df.csv')
		adapt_op_df.to_csv('adapt_op_df.csv')
		adapt_E_df.to_csv('adapt_E_df.csv')
		adapt_grad_df.to_csv('adapt_grad_df.csv')

	if enable_roto_2:
		adapt_roto_2_data_df = pd.DataFrame(adapt_roto_2_data_dict)
		adapt_roto_2_param_df = pd.DataFrame(adapt_roto_2_param_dict)
		adapt_roto_2_op_df = pd.DataFrame(adapt_roto_2_op_dict)
		adapt_roto_2_E_df = pd.DataFrame(adapt_roto_2_E_dict)
		adapt_roto_2_c_df = pd.DataFrame(adapt_roto_2_counter_dict)


		adapt_roto_2_data_df.to_csv('adapt_roto_2_data_df_{}.csv'.format(up))
		adapt_roto_2_param_df.to_csv('adapt_roto_2_param_df_{}.csv'.format(up))
		adapt_roto_2_op_df.to_csv('adapt_roto_2_op_df_{}.csv'.format(up))
		adapt_roto_2_E_df.to_csv('adapt_roto_2_E_df_{}.csv'.format(up))
		adapt_roto_2_c_df.to_csv('adapt_roto_2_c_df_{}.csv'.format(up))

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
	if enable_roto_2:
			out_file.write("ADAPTROTO with postprocessing enabled\n")

out_file.close()










