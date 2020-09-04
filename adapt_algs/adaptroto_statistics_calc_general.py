from ROTOADAPT_file_pauli_general import ROTOADAPTVQE, general_ROTOADAPT_OperatorSelector_pauli, split_into_paulis
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
#from Ha_max_adapt import ADAPT_maxH
from qiskit import IBMQ
import psutil
import sys
from qiskit.chemistry.components.initial_states import HartreeFock
from Super_opt_new import SuperBFGS_Grad, SuperL_BFGS_B
from bfgs_grad_new import BFGS_Grad
#from ShortcutADAPT_new import ShortcutADAPTVQE
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


up = int(sys.argv[1])
#up = 1

def retrieve_ham(number):
	adapt_data_df = pd.read_csv('load_adapt_data_df.csv')
	adapt_data_dict = adapt_data_df.to_dict()
	Ham_list = adapt_data_dict['hamiltonian']

	Ham = Ham_list[number]
	single_ham_list = Ham.split('\n')
	pauli_list = [0]*(len(single_ham_list)-1)
	weight_list = [0]*(len(single_ham_list)-1)
	for counter2 in range(1, len(single_ham_list)-1,1):
		print(single_ham_list[counter2][:4])
		pauli_list[counter2] = Pauli.from_label(single_ham_list[counter2][:4])
		weight_list[counter2] = complex(single_ham_list[counter2][6:-1])
	pauli_list[0] = Pauli.from_label('IIII')
	qubit_op = WeightedPauliOperator.from_list(pauli_list,weight_list)

	return qubit_op


import numpy as np
import pandas as pd
import scipy
import math
import datetime
import time
from qiskit.aqua import aqua_globals, QuantumInstance

import warnings
warnings.simplefilter("ignore")

starttime = datetime.datetime.now()


backend = Aer.get_backend('statevector_simulator')

shots = 1 #doesn't matter for statevector simulator 

qi = QuantumInstance(backend, shots)

output_to_file = 1
output_to_cmd = 1
store_in_df = 1
output_to_csv = 1
enable_adapt = 1
enable_roto_2 = 0
num_optimizer_runs = 100000

print('num available cpus', len(psutil.Process().cpu_affinity()))
print(starttime)
number_runs = 4
max_iterations = 5
ADAPT_stopping_gradient = 0 #not used
ADAPTROTO_stopping_energy = 0 #not used
ROTOSOLVE_stopping_energy = 1e-12
ADAPT_optimizer_stopping_energy = 1e-12
ROTOSOLVE_max_iterations = 100000

out_file = open("ADAPT_ROTO_RUN_INFO_{}.txt".format(up),"w+")

optimizer_name = "L_BFGS_B"
_num_restarts = 4
maxfun = 35000
maxiter = 35000
factr = 1
pgtol = 5e-16
superop = SuperL_BFGS_B(_num_restarts = 20, maxfun = maxfun, maxiter = maxiter, factr = 1)
#optimizer = BFGS_Grad(maxfun = maxfun, maxiter = maxiter, factr = 1, pgtol = pgtol)
#optimizer = SuperL_BFGS_B(_num_restarts = _num_restarts, maxfun = maxfun, maxiter = maxiter, factr = 1)
mini_optimizer = L_BFGS_B(maxfun = maxfun, maxiter = maxiter, factr = 1)
optimizer = SuperL_BFGS_B(_num_restarts = 15, maxfun = maxfun, maxiter = maxiter, factr = 1)

it=HartreeFock(num_qubits=4,num_orbitals=6,num_particles=2,two_qubit_reduction=True,qubit_mapping='parity')
#it = None
#optimizer = Rotosolve(ROTOSOLVE_stopping_energy,ROTOSOLVE_max_iterations, param_per_step = 2)

adapt_data_dict = {'hamiltonian': [], 'eval time': [], 'num op choice evals': [], 'num optimizer evals': [], 'ansz length': [], 'final energy': []}
adapt_param_dict = dict()
adapt_op_dict = dict()
adapt_E_dict = dict()
adapt_info_dict = dict()
adapt_grad_dict = dict()
adapt_echange_dict = dict()
adapt_Ha_dict = dict()
adapt_A_dict = dict()



adapt_roto_2_data_dict = {'hamiltonian': [], 'eval time': [], 'num optimizer evals': [], 'num op choice evals': [], 'ansz length': [], 'final energy': []}
adapt_roto_2_param_dict = dict()
adapt_roto_2_op_dict = dict()
adapt_roto_2_E_dict = dict()
adapt_roto_2_info_dict = dict()
adapt_roto_2_grad_dict = dict()
adapt_roto_2_echange_dict = dict()
adapt_roto_2_Ha_dict = dict()
adapt_roto_2_A_dict = dict()



Exact_energy_dict = {'ground energy':[]}
num_qubits = 4
counter_start = 0
counter = counter_start

distance =[0.5,0.75,1,1.25,1.5]

num_term_list = [50,100,150,200]
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


def quaternary (n): #this function taken from stackexchange
	"""
		converts any base 10 number to base 3
		Is used in this case to convert the "nonzero term list" entry to base 3
		as a base 3 representation can be used to reconstruct the sine and cosine constant multiples
	"""
	if n == 0:
		return '0'
	nums = []
	while n:
		n, r = divmod(n, 4)
		nums.append(str(r))
	return ''.join(reversed(nums))

while counter <= (number_runs + counter_start - 1):
	if counter == 0:
		op_selec = "A max"
	elif counter == 1:
		op_selec = "Hc min"
	elif counter == 2:
		op_selec = "Ha max"
	else:
		op_selec = "energy max"
	#max_iterations = counter + 1
	#num_terms = num_term_list[counter]
	#mat = np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits))
	#mat = scipy.sparse.random(2**num_qubits, 2**num_qubits, density = 0.5) + 1j*scipy.sparse.random(2**num_qubits, 2**num_qubits, density = 0.5)
	#mat = np.conjugate(np.transpose(scipy.sparse.csr_matrix.todense(mat))) + scipy.sparse.csr_matrix.todense(mat)
	#mat = np.conjugate(np.transpose(mat)) + mat
	#ham = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"
	#ham = ham + 0.2*(counter+2)*Gen_rand_1_ham(1,num_qubits)
	#dist in distances = np.arange(0.5, 4.0, 0.1) or could do 2A
	dist = 1.5
	ham, num_particles, num_spin_orbitals, shift = get_qubit_op(dist)
	#ham  = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
	#big_list = [0]*4**num_qubits
	#for i in range(0,num_terms):
	#	num = np.random.randint(1,4**num_qubits)
	#	if big_list[num] == 0:
	#		qnum = quaternary(num)
	#		ham = ham + WeightedPauliOperator.from_list([Pauli.from_label(create_term(qnum, num_qubits))], [np.random.uniform(-1.5,1.5)])
	#		big_list[num] = 1
	#print(ham.print_details())
	#ham = retrieve_ham(counter)
	#ham = random_diagonal_hermitian(num_qubits)
	#print(ham.print_details())
	#ham = get_h_4_hamiltonian(counter*0.25 + 0.25, 2, "jw")
	#ham = retrieve_ham(counter)
	
	qubit_op = ham
	num_qubits = qubit_op.num_qubits

	print('num qubits', qubit_op.num_qubits)
	start = time.time()
	##pool = CompletePauliPool.from_num_qubits(num_qubits)
	#if up == 1:
	#	pool = PauliPool.from_pauli_strings(['IYIX', 'ZXYI', 'XZYI', 'IZXY', 'IIYI', 'IZYI'])
	#	print(['IYIX', 'ZXYI', 'XZYI', 'IZXY', 'IIYI', 'IZYI'])
	#	pool_name = ['IYIX', 'ZXYI', 'XZYI', 'IZXY', 'IIYI', 'IZYI']
	#else:
	#	pool = PauliPool.from_pauli_strings(['ZXYI', 'XYII', 'XIYI', 'IXZY', 'IIXY', 'IZYI'])
	#	print(['ZXYI', 'XYII', 'XIYI', 'IXZY', 'IIXY', 'IZYI'])
	#	pool_name = ['ZXYI', 'XYII', 'XIYI', 'IXZY', 'IIXY', 'IZYI']
	#if up >= 0:
	#	conn = 3
	#elif up > 20:
	#	conn = 4
	#else:
	#	conn = 2
	#seed = []
	#pool = []
	#print('connectivity', conn)
	#while not pool:
	#	seed = []
	#	for i in range(0,num_qubits):
	#		seed.append(str(np.random.randint(0,4)))
	#	seed = create_term(seed, num_qubits)
	#	pool = ConjectureBasedSubset(conn, seed)
	#print(seed)
	#pool_name = pool[np.random.randint(0,len(pool)-1)]
	#print(pool_name)
	#pool = PauliPool.from_pauli_strings(pool_name)
	pool = PauliPool.from_all_pauli_strings(num_qubits) #all possible pauli strings
	#s1 = SeparableInitialStateReal(qubit_op,superop)
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
	#if enable_adapt and store_in_df:
		#adapt_data_dict['pool_name'].append(pool_name)
	#if enable_roto_2 and store_in_df:
		#adapt_roto_2_data_dict['pool_name'].append(pool_name)
	#for op in pool.pool:
	#	print(op.print_details())


	gentime = time.time() - start
	print('done generating pool', gentime)
	Exact_result = ExactEigensolver(qubit_op).run()
	print('energies', Exact_result['energies'])
	Exact_energy_dict['ground energy'].append(Exact_result['energy'])
	if output_to_file:
		out_file.write("Exact Energy: {}\n".format(Exact_result['energy']))

	if enable_adapt:
		print('on adapt')
		adapt = ROTOADAPTVQE(operator_pool=pool, initial_state = it, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations, energy_tol = ADAPT_stopping_gradient, initial_parameters = 3, operator_selector = general_ROTOADAPT_OperatorSelector_pauli(qubit_op, operator_pool=pool, drop_duplicate_circuits=True, energy_tol = ADAPTROTO_stopping_energy, op_mode = "adapt", split_sets = False), shortcut = True)
		start = time.time()
		adapt_result = adapt.run(qi)
		eval_time = time.time() - start
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		info_list = []
		param_list = []
		grad_list = []
		echange_list = []
		Ha_list = []
		A_list = []
		for i in range(0,(len(adapt_result))):
			num_op_evals = num_op_evals + adapt_result[i]['eval_count']
			num_op_choice_evals = num_op_choice_evals + adapt_result[i]['num op choice evals']
			energy_history.append(adapt_result[i]['energy'])
			op_list.append(adapt_result[-1]['current_ops'][i].print_details())
			param_list.append(adapt._step_history[i]['opt_params'])
			info_list.append([adapt._step_history[i]['grad info'], adapt._step_history[i]['energy info'],  adapt._step_history[i]['A info'],  adapt._step_history[i]['Hc info'],  adapt._step_history[i]['Ha info']])
			grad_list.append(adapt._step_history[i]['grads'])
			echange_list.append(adapt._step_history[i]['energy change'])
			Ha_list.append(adapt._step_history[i]['Ha list'])
			A_list.append(adapt._step_history[i]['A list'])
		print(energy_history)


		if output_to_cmd:
			#print("ADAPT ROTO 2 Results for \n{}".format(ham.print_details()))
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

			adapt_param_dict.update({'Ham_{}'.format(counter): param_list})
			adapt_op_dict.update( {'Ham_{}'.format(counter): op_list})
			adapt_E_dict.update({'Ham_{}'.format(counter): energy_history})
			adapt_info_dict.update({'Ham_{}'.format(counter): info_list})
			adapt_echange_dict.update({'Ham_{}'.format(counter): echange_list})
			adapt_grad_dict.update({'Ham_{}'.format(counter): grad_list})
			adapt_Ha_dict.update({'Ham_{}'.format(counter): Ha_list})
			adapt_A_dict.update({'Ham_{}'.format(counter): A_list})

	if enable_roto_2:
		print('on roto 2')
		adapt_roto_2 = ROTOADAPTVQE(operator_pool=pool, initial_state = it, vqe_optimizer=optimizer, hamiltonian=qubit_op, max_iters = max_iterations, energy_tol = ADAPT_stopping_gradient, initial_parameters = 1, operator_selector = general_ROTOADAPT_OperatorSelector_pauli(qubit_op, operator_pool=pool, drop_duplicate_circuits=True, mini_optimizer = mini_optimizer, energy_tol = ADAPTROTO_stopping_energy, op_mode = "energy", split_sets = False, parameters_per_step = 1, two_op_mode = False), shortcut = False)
		start = time.time()
		adapt_roto_2_result = adapt_roto_2.run(qi)
		eval_time = time.time() - start
		num_op_evals = 0
		num_op_choice_evals = 0
		energy_history = []
		op_list = []
		info_list = []
		param_list = []
		grad_list = []
		echange_list = []
		Ha_list = []
		A_list = []
		roto_param_list = []
		for i in range(0,(len(adapt_roto_2_result))):
			num_op_evals = num_op_evals + adapt_roto_2_result[i]['eval_count']
			num_op_choice_evals = num_op_choice_evals + adapt_roto_2_result[i]['num op choice evals']
			energy_history.append(adapt_roto_2_result[i]['energy'])
			op_list.append(adapt_roto_2_result[-1]['current_ops'][i].print_details())
			param_list.append(adapt_roto_2._step_history[i]['opt_params'])
			info_list.append([adapt_roto_2._step_history[i]['grad info'], adapt_roto_2._step_history[i]['energy info'], adapt_roto_2._step_history[i]['A info'], adapt_roto_2._step_history[i]['Hc info'], adapt_roto_2._step_history[i]['Ha info']])
			grad_list.append(adapt_roto_2._step_history[i]['grads'])
			echange_list.append(adapt_roto_2._step_history[i]['energy change'])
			Ha_list.append(adapt_roto_2._step_history[i]['Ha list'])
			A_list.append(adapt_roto_2._step_history[i]['A list'])
			roto_param_list.append(adapt_roto_2._step_history[i]['optimal param'])
		print(energy_history)

		if output_to_cmd:
			#print("ADAPT ROTO 2 Results for \n{}".format(ham.print_details()))
			print("Total Eval Time", eval_time)
			print("total number of op evaluations", num_op_evals)
			print("total number of op choice evals", num_op_choice_evals)
			print("ansatz length", len(adapt_roto_2_result))
			print("optimal parameters", adapt_roto_2_result[-1]['opt_params'])
			print("operator list", op_list)
			print("energy history", energy_history)
			print("roto param list", roto_param_list)

		if store_in_df:
			adapt_roto_2_data_dict['hamiltonian'].append(ham.print_details())
			adapt_roto_2_data_dict['eval time'].append(eval_time)
			adapt_roto_2_data_dict['num optimizer evals'].append(num_op_evals)
			adapt_roto_2_data_dict['num op choice evals'].append(num_op_choice_evals)
			adapt_roto_2_data_dict['ansz length'].append(len(adapt_roto_2_result))
			adapt_roto_2_data_dict['final energy'].append(energy_history[-1])

			adapt_roto_2_param_dict.update({'Ham_{}'.format(counter): param_list})
			adapt_roto_2_op_dict.update( {'Ham_{}'.format(counter): op_list})
			adapt_roto_2_E_dict.update({'Ham_{}'.format(counter): energy_history})
			adapt_roto_2_info_dict.update({'Ham_{}'.format(counter): info_list})
			adapt_roto_2_echange_dict.update({'Ham_{}'.format(counter): echange_list})
			adapt_roto_2_grad_dict.update({'Ham_{}'.format(counter): grad_list})
			adapt_roto_2_Ha_dict.update({'Ham_{}'.format(counter): Ha_list})
			adapt_roto_2_A_dict.update({'Ham_{}'.format(counter): A_list})

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
		adapt_info_df = pd.DataFrame(adapt_info_dict)
		adapt_echange_df = pd.DataFrame(adapt_echange_dict)
		adapt_grad_df = pd.DataFrame(adapt_grad_dict)
		adapt_Ha_df = pd.DataFrame(adapt_Ha_dict)
		adapt_A_df = pd.DataFrame(adapt_A_dict)

		adapt_data_df.to_csv('adapt_data_df_{}.csv'.format(up))
		adapt_param_df.to_csv('adapt_param_df_{}.csv'.format(up))
		adapt_op_df.to_csv('adapt_op_df_{}.csv'.format(up))
		adapt_E_df.to_csv('adapt_E_df_{}.csv'.format(up))
		adapt_info_df.to_csv('adapt_op_selec_info_df_{}.csv'.format(up))
		adapt_echange_df.to_csv('adapt_echange_df_{}.csv'.format(up))
		adapt_grad_df.to_csv('adapt_grad_df_{}.csv'.format(up))
		adapt_Ha_df.to_csv('adapt_Ha_df_{}.csv'.format(up))
		adapt_A_df.to_csv('adapt_A_df_{}.csv'.format(up))

	if enable_roto_2:
		adapt_roto_2_data_df = pd.DataFrame(adapt_roto_2_data_dict)
		adapt_roto_2_param_df = pd.DataFrame(adapt_roto_2_param_dict)
		adapt_roto_2_op_df = pd.DataFrame(adapt_roto_2_op_dict)
		adapt_roto_2_E_df = pd.DataFrame(adapt_roto_2_E_dict)
		adapt_roto_2_info_df = pd.DataFrame(adapt_roto_2_info_dict)
		adapt_roto_2_echange_df = pd.DataFrame(adapt_roto_2_echange_dict)
		adapt_roto_2_grad_df = pd.DataFrame(adapt_roto_2_grad_dict)
		adapt_roto_2_Ha_df = pd.DataFrame(adapt_roto_2_Ha_dict)
		adapt_roto_2_A_df = pd.DataFrame(adapt_roto_2_A_dict)

		adapt_roto_2_data_df.to_csv('adapt_roto_2_data_df_{}.csv'.format(up))
		adapt_roto_2_param_df.to_csv('adapt_roto_2_param_df_{}.csv'.format(up))
		adapt_roto_2_op_df.to_csv('adapt_roto_2_op_df_{}.csv'.format(up))
		adapt_roto_2_E_df.to_csv('adapt_roto_2_E_df_{}.csv'.format(up))
		adapt_roto_2_info_df.to_csv('adapt_roto_op_selec_info_df_{}.csv'.format(up))
		adapt_roto_2_echange_df.to_csv('adapt_roto_echange_df_{}.csv'.format(up))
		adapt_roto_2_grad_df.to_csv('adapt_roto_grad_df_{}.csv'.format(up))
		adapt_roto_2_Ha_df.to_csv('adapt_roto_Ha_df_{}.csv'.format(up))
		adapt_roto_2_A_df.to_csv('adapt_roto_A_df_{}.csv'.format(up))

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





