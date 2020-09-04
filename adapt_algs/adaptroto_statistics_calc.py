from ROTOADAPT_file_pauli_general import ROTOADAPTVQE, general_ROTOADAPT_OperatorSelector_pauli
from adapt_new import ADAPTVQE
#from rotosolve_edited import Rotosolve
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool, CompletePauliPool
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
#from operator_selector_new import AntiCommutingSelector
from mol_ham_file import get_qubit_op
from Ha_max_adapt import ADAPT_maxH
#from CVQE import split_into_paulis
from qiskit import IBMQ
import psutil
import sys
from qiskit.chemistry.components.initial_states import HartreeFock
from Super_opt_new import SuperBFGS_Grad, SuperL_BFGS_B
from bfgs_grad_new import BFGS_Grad
from p_bfgs_new import P_BFGS
from ShortcutADAPT_new import ShortcutADAPTVQE
from qiskit.aqua.components.initial_states import Custom
from Separable_initial_state_new import SeparableInitialStateReal
from Pool_generator import ConjectureBasedSubset
from qisresearch.i_vqe.vis import plot_adapt_dashboard
#from CADAPTVQE import CADAPTVQE
#from Generate_rand_equal_ham import Gen_rand_1_ham


def create_term(num,num_qubits):
	new_string = ''
	if len(num) < num_qubits:
		for i in range(0,num_qubits-len(num)):
			new_string = new_string + 'I'
	for i in range(0,len(num)):
		if num[i] == '0':
			new_string = new_string + 'I'
		elif num[i] == '1':
			new_string = new_string + 'X'
		elif num[i] == '2':
			new_string = new_string + 'Y'
		else:
			new_string = new_string + 'Z'
	return new_string


up = int(sys.argv[1]) - 1

def retrieve_ham(number):
	adapt_data_df = pd.read_csv('load_adapt_data_df.csv')
	adapt_data_dict = adapt_data_df.to_dict()
	Ham_list = adapt_data_dict['hamiltonian']

	Ham = Ham_list[number]
	single_ham_list = Ham.split('\n')
	pauli_list = [0]*(len(single_ham_list)-1)
	weight_list = [0]*(len(single_ham_list)-1)
	for counter2 in range(0, len(single_ham_list)-1,1):
		pauli_list[counter2] = Pauli.from_label(single_ham_list[counter2][:4])
		weight_list[counter2] = complex(single_ham_list[counter2][6:-1])
	qubit_op = WeightedPauliOperator.from_list(pauli_list,weight_list)

	return qubit_op


import numpy as np
import pandas as pd
import scipy
import math
import datetime
import time
from qiskit.aqua import aqua_globals, QuantumInstance
import psutil

import warnings
warnings.simplefilter("ignore")

starttime = datetime.datetime.now()


output_to_file = 1
output_to_cmd = 1
store_in_df = 1
output_to_csv = 1
enable_adapt = 0
enable_roto_2 = 1
num_optimizer_runs = 100000

print('num available cpus', len(psutil.Process().cpu_affinity()))
print(starttime)
number_runs = 1
max_iterations = 20
ADAPT_stopping_gradient = 0 #not used
ADAPTROTO_stopping_energy = 0 #not used
ROTOSOLVE_stopping_energy = 1e-12
ADAPT_optimizer_stopping_energy = 1e-12
ROTOSOLVE_max_iterations = 100000

out_file = open("Run_info_{}.txt".format(up),"w+")

optimizer_name = "Super_BFGS_Grad"
_num_restarts = 10 #now included in algorithm itself.
maxfun = 20000
maxiter = 20000
factr = 10
pgtol = 1e-15
#optimizer = SuperBFGS_Grad(_num_restarts = _num_restarts, maxfun = maxfun, maxiter = maxiter, factr = 1, pgtol = pgtol)
mini_optimizer = L_BFGS_B(maxfun = maxfun, maxiter = maxiter, factr = factr)
#optimizer = BFGS_Grad(maxfun = maxfun, maxiter = maxiter, factr = factr, pgtol = pgtol)
superoptimizer = SuperL_BFGS_B(maxfun = maxfun, maxiter = maxiter, factr = factr, _num_restarts = _num_restarts)
#optimizer = P_BFGS(maxfun = maxfun, factr = factr, max_processes = 3)
optimizer = mini_optimizer

backend = Aer.get_backend('statevector_simulator')
backend_options = {'max_parallel_threads': len(psutil.Process().cpu_affinity()), 'max_parallel_experiments': 0}
shots = 1
qi = QuantumInstance(backend, shots = shots)
qi.set_config(**backend_options)


it=HartreeFock(num_qubits=4,num_orbitals=6,num_particles=2,two_qubit_reduction=True,qubit_mapping='parity')
#it = None
#optimizer = Rotosolve(ROTOSOLVE_stopping_energy,ROTOSOLVE_max_iterations, param_per_step = 2)

adapt_data_dict = {'hamiltonian': [], 'eval time': [], 'num op choice evals': [], 'num optimizer evals': [], 'ansz length': [], 'final energy': []}
adapt_param_dict = dict()
adapt_op_dict = dict()
adapt_E_dict = dict()
adapt_counter_dict = dict()


adapt_roto_2_data_dict = {'hamiltonian': [], 'eval time': [], 'num optimizer evals': [], 'num op choice evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_2_param_dict = dict()
adapt_roto_2_op_dict = dict()
adapt_roto_2_E_dict = dict()
adapt_roto_2_counter_dict = dict()


Exact_energy_dict = {'ground energy':[]}
num_qubits = 6
counter_start = 0
counter = counter_start

distance =[0.5,0.75,1,1.25,1.5]

while counter <= (number_runs + counter_start - 1):

	#mat = np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits))
	#mat = scipy.sparse.random(2**num_qubits, 2**num_qubits, density = 0.5) + 1j*scipy.sparse.random(2**num_qubits, 2**num_qubits, density = 0.5)
	#mat = np.conjugate(np.transpose(scipy.sparse.csr_matrix.todense(mat))) + scipy.sparse.csr_matrix.todense(mat)
	#mat = np.conjugate(np.transpose(mat)) + mat
	#ham = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"
	#ham = ham + 0.2*(counter+2)*Gen_rand_1_ham(1,num_qubits)
	#dist in distances = np.arange(0.5, 4.0, 0.1) or could do 2A
	dist = 1.5
	#dist = 1
	ham, num_particles, num_spin_orbitals, shift = get_qubit_op(dist)
	#print(ham.print_details())
	#ham = random_diagonal_hermitian(num_qubits)
	#print(ham.print_details())
	#ham = get_h_4_hamiltonian(dist, 2, "jw")
	#it = get_h_4_hamiltonian(dist, 2, "jw", return_hf_state = True)
	#ham = retrieve_ham(counter)
	qubit_op = ham
	num_qubits = qubit_op.num_qubits

	print('num qubits', qubit_op.num_qubits)
	start = time.time()
	#pool = CompletePauliPool.from_num_qubits(num_qubits)
	#conn = 2
	#seed = []
	#pool = []
	#while not pool:
	#	seed = []
	#	for i in range(0,num_qubits):
	#		seed.append(str(np.random.randint(0,3)))
	#	seed = create_term(seed, num_qubits)
	#	pool = ConjectureBasedSubset(conn, seed)
	#print(seed)
	#print(pool[np.random.randint(0,len(pool)-1)])
	#pool = PauliPool.from_pauli_strings(pool[np.random.randint(0,len(pool)-1)])
	pool = PauliPool.from_all_pauli_strings(num_qubits) #all possible pauli strings
	#s1 = SeparableInitialStateReal(qubit_op,superoptimizer)
	#sout = s1.initialize(qi)
	#it = s1
	#it = Custom(qubit_op.num_qubits, state = 'uniform')
	#pool = PauliPool.from_pauli_strings(['YXII','IYXI','IIYX','IYIX','IIIY','IIYI'])
	#pool = PauliPool()
	#pool._num_qubits = num_qubits
	#pool._pool = generate_lie_algebra(ham)
	#pool.cast_out_even_y_paulis(True) #more efficient pool
	#pool.cast_out_higher_order_z_strings(True)
	#pool.cast_out_particle_number_violating_strings(True)
	gentime = time.time() - start
	print('done generating pool', gentime)
	Exact_result = ExactEigensolver(qubit_op, 16).run()
	print('energies', Exact_result['energies'])
	Exact_energy_dict['ground energy'].append(Exact_result['energy'])
	if output_to_file:
		out_file.write("Exact Energy: {}\n".format(Exact_result['energy']))

	if enable_adapt:
		print('on adapt')
		adapt = ROTOADAPTVQE(operator_pool=pool, initial_state = it, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations, energy_tol = ADAPT_stopping_gradient, initial_parameters = 1, operator_selector = general_ROTOADAPT_OperatorSelector_pauli(qubit_op, operator_pool=pool, drop_duplicate_circuits=True, energy_tol = ADAPTROTO_stopping_energy, op_mode = "adapt", split_sets = False), shortcut = False)
		start = time.time()
		adapt_result = adapt.run(qi)
		eval_time = time.time() - start
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		counter_list = []
		for i in range(0,(len(adapt_result))):
			num_op_evals = num_op_evals + adapt_result[i]['eval_count']
			num_op_choice_evals = num_op_choice_evals + adapt_result[i]['num op choice evals']
			energy_history.append(adapt_result[i]['energy'])
			op_list.append(adapt_result[-1]['current_ops'][i].print_details())
			counter_list.append(adapt_result[i]['optimizer_counter'])
		print(energy_history)


		if output_to_cmd:
			print("ADAPT ROTO 2 Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", eval_time)
			print("total number of op evaluations", num_op_evals)
			print("total number of op choice evals", num_op_choice_evals)
			print("ansatz length", len(adapt_result))
			print("optimal parameters", adapt_result[-1]['opt_params'])
			print("operator list", op_list)
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
			adapt_counter_dict.update({'Ham_{}'.format(counter): counter_list})

		plot_adapt_dashboard(adapt._step_history, qubit_op, True)

	if enable_roto_2:
		print('on roto 2')
		adapt_roto_2 = ROTOADAPTVQE(operator_pool=pool, initial_state = it, vqe_optimizer=optimizer,
		 hamiltonian=qubit_op, max_iters = max_iterations, energy_tol = ADAPT_stopping_gradient,
		 initial_parameters = 1, operator_selector = general_ROTOADAPT_OperatorSelector_pauli(qubit_op, operator_pool=pool, drop_duplicate_circuits=True, energy_tol = ADAPTROTO_stopping_energy, op_mode = "energy", parameters_per_step = 1, mini_optimizer = mini_optimizer, two_op_mode = False, split_sets = False))
		start = time.time()
		adapt_roto_2_result = adapt_roto_2.run(qi)
		eval_time = time.time() - start
		print(adapt_roto_2._step_history)
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


			plot_adapt_dashboard(adapt_roto_2._step_history, qubit_op, True)

	counter += 1
	print("time", datetime.datetime.now())
	print("counter", counter)


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
		adapt_counter_df = pd.DataFrame(adapt_counter_dict)

		adapt_data_df.to_csv('adapt_data_df_{}.csv'.format(up))
		adapt_param_df.to_csv('adapt_param_df_{}.csv'.format(up))
		adapt_op_df.to_csv('adapt_op_df_{}.csv'.format(up))
		adapt_E_df.to_csv('adapt_E_df_{}.csv'.format(up))
		adapt_counter_df.to_csv('adapt_counter_df_{}.csv'.format(up))

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
			out_file.write("num restarts: {}".format(_num_restarts))
			out_file.write("maxfun: {}".format(maxfun))
			out_file.write("maxiter: {}".format(maxiter))
			out_file.write("factr: {}".format(factr))
			out_file.write("pgtol: {}".format(pgtol))
	if enable_roto_2:
			out_file.write("ADAPTROTO with postprocessing enabled\n")

out_file.close()



"""
	if enable_adapt:
		print('on adapt')
		adapt_vqe = ShortcutADAPTVQE(pool=pool, initial_state=it, optimizer=optimizer, operator=qubit_op, max_iters = max_iterations, grad_tol = ADAPT_stopping_gradient)
		start = time.time()
		result = adapt_vqe.run(qi)
		adapt_result = adapt_vqe.adapt_step_history
		eval_time = time.time() - start
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		grad_list = []
		print(adapt_result)
		print(result)
		#for i in range(0,(len(adapt_result))):
		#	num_op_evals = 0#num_op_evals + adapt_result[i]['eval_count']
		#	num_op_choice_evals = 0#num_op_choice_evals + adapt_result[i]['num op choice evals']
		#	energy_history.append(adapt_result[i]['energy'])
		#	op_list.append(adapt_result[-1]['current_ops'][i].print_details())
		#	grad_list.append(adapt_result[i]['max_grad'])

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
"""




"""
	if enable_adapt:
		print('on adapt')
		#AntiCommutingSelector(hamiltonian = qubit_op, operator_pool = pool, drop_duplicate_circuits = True, grad_tol = ADAPT_stopping_gradient)
		adapt_vqe = ADAPT_maxH(operator_pool=pool, initial_state=it, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations)
		start = time.time()
		adapt_result = adapt_vqe.run(qi)
		eval_time = time.time() - start
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
"""

"""
	if enable_adapt:
		print('on adapt')
		adapt_vqe = ROTOADAPTVQE(operator_pool=pool, initial_state = it, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations, energy_tol = ADAPT_stopping_gradient, initial_parameters = 0, operator_selector = general_ROTOADAPT_OperatorSelector_pauli(qubit_op, operator_pool=pool, drop_duplicate_circuits=True, energy_tol = ADAPTROTO_stopping_energy, op_mode = "A max"))
		start = time.time()
		adapt_result = adapt_vqe.run(qi)
		print(adapt_result)
		eval_time = time.time() - start
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
			grad_list.append(adapt_result[i]['grad info'][0])

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
"""





