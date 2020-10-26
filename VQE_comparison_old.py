from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.algorithms import VQE
import numpy as np
import time
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import SLSQP, NELDER_MEAD
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from CVQE_serial import Classical_VQE as CVQE
#from CVQE import find_commutator, split_into_paulis
from qiskit.quantum_info import Pauli
from qisresearch.adapt.adapt_variational_form import ADAPTVariationalForm
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.components.initial_states import InitialState, Zero
from qisresearch.utils.random_hamiltonians import random_diagonal_hermitian, get_h_4_hamiltonian
from qiskit.aqua.operators.op_converter import to_matrix_operator
#from Generate_rand_equal_ham import Gen_rand_1_ham, Gen_rand_rand_ham, random_pauli
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

"""
outfi = open("VQE_comparison.txt","w",1)
outfi.write("")
"""
def reverse(lst): 
    lst.reverse() 
    return lst 

def retrieve_ham(number, num_qubits = 4):
    adapt_data_df = pd.read_csv('load_adapt_roto_2_data_df.csv')
    adapt_data_dict = adapt_data_df.to_dict()
    Ham_list = adapt_data_dict['hamiltonian']

    Ham = Ham_list[number]
    single_ham_list = Ham.split('\n')
    pauli_list = [0]*(len(single_ham_list)-1)
    weight_list = [0]*(len(single_ham_list)-1)
    for counter2 in range(0, len(single_ham_list)-1,1):
        pauli_list[counter2] = Pauli.from_label(single_ham_list[counter2][:num_qubits])
        weight_list[counter2] = complex(single_ham_list[counter2][num_qubits:])
    qubit_op = WeightedPauliOperator.from_list(pauli_list,weight_list)

    return qubit_op

def retrieve_op_list(number, rev = False, num_qubits = 4):
    adapt_op_df = pd.read_csv('load_adapt_roto_2_op_df.csv')
    adapt_op_dict = adapt_op_df.to_dict()
    op_label_list = list(adapt_op_dict['Ham_{}'.format(number)])
    op_list = []
    for label in op_label_list:
        op_name = adapt_op_dict['Ham_{}'.format(number)][label]
        print(op_name)
        op = Pauli.from_label(op_name[:num_qubits])
        op = WeightedPauliOperator.from_list([op])
        op_list.append(op)
    if rev:
        return reverse(op_list)
    return op_list

backend = Aer.get_backend("statevector_simulator")
qi = QuantumInstance(backend)
optimizer = NELDER_MEAD(tol = 1e-10)

ansatz_length = 5
op_list = []
"""
"""

def be_in_range(a, b, n):
    l = n
    while l > b:
        l = l-(b-a)
    while l < a:
        l = l + (b-a)
    if l>b:
        return error 
    else:
        return l


runtime_sum = 0
exact_time_sum = 0
runs = 1
k = 0
i = 0
runtime_array = []
vqeruntime_array = []
exact_time_array = []
cvqe_dict = {'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}
vqe_dict = {'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}
exact_dict = {'2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], '10': []}
num_qubits = 4
#test = random_diagonal_hermitian(2)
#print(test.print_details())
#mat = to_matrix_operator(test)
#print(mat.print_details())
for i in range(0,1):
    k = 0
    runtime_sum = 0
    exact_time_sum = 0
    vqeruntime_sum = 0
    print('num_qubits', num_qubits)
    while k < runs:
        initial_state = Zero(num_qubits)
        op_list = []
        wop = 0*WeightedPauliOperator.from_list(paulis=[Pauli.from_label('I'*num_qubits)], weights=[1.0])
        #mat = np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits)) + 1j * np.random.uniform(0, 1, size=(2**num_qubits, 2**num_qubits))
        #mat = np.conjugate(np.transpose(mat)) + mat
        
        #ham = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"
        ham = get_h_4_hamiltonian(0.25, 2, "jw")
        #ham = retrieve_ham(0, num_qubits = num_qubits)
        print('old_ham', ham.print_details())
        #ham = get_h_4_hamiltonian(0.5,2,'jw')
        #op_list = retrieve_op_list(0, rev = True, num_qubits = num_qubits)
        #for op in op_list:
        #    print(op.print_details())
        op_list = [WeightedPauliOperator.from_list([Pauli.from_label('IIXX')]), WeightedPauliOperator.from_list([Pauli.from_label('XXXY')])]
        #op_list = [WeightedPauliOperator.from_list([Pauli.from_label('YYZZ')],[1.0]),
        # WeightedPauliOperator.from_list([Pauli.from_label('XXXY')], [1.0]),
        # WeightedPauliOperator.from_list([Pauli.from_label('IIYZ')], [1.0])]
        print(ham.print_details())
        #ham = Gen_rand_rand_ham(k+1, num_qubits)
        qubitOp = ham
        num_qubits = qubitOp.num_qubits
        print('iteration: ', k+1)
        #for i in range(0,ansatz_length):
        #    op_list.append(WeightedPauliOperator.from_list([random_pauli(num_qubits)], [1.0]))
        #    print(op_list[i].print_details())
        #ham = get_h_4_hamiltonian(1.5, 1, 'jw')

        var_form = ADAPTVariationalForm(operator_pool = op_list, bounds = None, initial_state = initial_state)
        print('started cvqe')
        #cvqe = CVQE(operator = qubitOp, optimizer = optimizer, operator_list = reverse(op_list), var_form = var_form, expec_mode = False, dir_to_bracket = True)
        cvqe2 = CVQE(operator = qubitOp, optimizer = optimizer, operator_list = op_list, var_form = var_form, expec_mode = False, dir_to_bracket = False, param_list = None)#[1.570796326,0.098254701,0.044596535,-0.043753889,-0.024820417,-0.024565925,0.022926426,-0.003359644,0.002535945,-0.003167673,0.00220233,0.000897351,0.00068846,-0.000336171,-0.000340844,-0.000170536,1.13E-06])
        #start = time.time()
        #cvqe_result = cvqe.run()
        #runtime = time.time()-start
        print('started cvqe 2')
        start = time.time()
        cvqe2_result = cvqe2.run(check_params = False)
        cvqe2runtime = time.time()-start
        print('ended cvqe')

        #vqe = VQE(qubitOp, var_form, optimizer)
        #start = time.time()
        #vqe_result = vqe.run(backend)
        #vqeruntime = time.time()-start
        #print('vqe ', vqe_result['min_val'])
        #for i in range(0, len(vqe_result['opt_params'])):
        #    result = be_in_range(0,np.pi,vqe_result['opt_params'][i])
        #    print('vqe params', result)
        #print('vqe time', vqeruntime)
        #vqeruntime_sum = vqeruntime_sum + vqeruntime
        #vqe_dict['{}'.format(num_qubits)].append(vqeruntime)
    
        #print('cvqe ', cvqe_result['min_val'])
        #for i in range(0,len(cvqe_result['opt_params'])):
        #    result = be_in_range(0,np.pi,cvqe_result['opt_params'][i])
        #    print('cvqe params', result)
        #print('cvqe time', runtime)

        print('cvqe 2', cvqe2_result['min_val'])
        for i in range(0,len(cvqe2_result['opt_params'])):
            result = be_in_range(0,np.pi,cvqe2_result['opt_params'][i])
            print('cvqe2  params', result)
        print('cvqe 2 time', cvqe2runtime)
        #runtime_sum = runtime + runtime_sum
        #cvqe_dict['{}'.format(num_qubits)].append(runtime)

        exact_stime = time.time()
        result = ExactEigensolver(qubitOp).run()
        print('exact', result['energy'])
        exact_time = time.time() - exact_stime
        exact_time_sum = exact_time_sum + exact_time
        #exact_dict['{}'.format(num_qubits)].append(exact_time)
        #print('exact ', result['energy'])
        #print('exact time', exact_time)
        k = k+1
        #print('cvqe moving avg', runtime_sum/(k+1))
        #print('vqe moving avg', vqeruntime_sum/(k+1))
        #print('exact moving avg', exact_time_sum/(k+1))

    #print('runtime avg: ', runtime_sum/runs)
    #print('exact_time_avg', exact_time_sum/runs)
    #print('vqe runtim avg: ', vqeruntime_sum/runs)
    runtime_array.append(runtime_sum/runs)
    #vqeruntime_array.append(vqeruntime_sum/runs)
    exact_time_array.append(exact_time_sum/runs)

#print('runtime_array', runtime_array)
#print('exacttime array', exact_time_array)
#print('vqe time array', vqeruntime_array)
"""
vqe_df = pd.DataFrame(vqe_dict)
vqe_df.to_csv('vqe_times.csv')
cvqe_df = pd.DataFrame(cvqe_dict)
cvqe_df.to_csv('cvqe_times.csv')
exact_df = pd.DataFrame(exact_dict)
exact_df.to_csv('exact_times.csv')
outfi.close()
"""