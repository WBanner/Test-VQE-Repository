"""
ADAPTROTO File
"""

import logging
from copy import deepcopy
from typing import List, Union, Dict
import pandas as pd
import sys
sys.path.append("usr/local/lib/python3.7/site-packages")

import numpy as np
from qiskit.aqua import AquaError
from qiskit import QuantumCircuit
from qiskit.aqua.components.initial_states import InitialState, Zero, Custom
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
from qiskit.quantum_info import Pauli
from qiskit.tools import parallel_map
from qisresearch.adapt.adapt_variational_form import ADAPTVariationalForm
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool
from qisresearch.vqa import DummyOptimizer
from qisresearch.i_vqe.callbacks import Callback
from iterative_new import IterativeVQE
from operator_selector_new import OperatorSelector, multi_circuit_eval, get_Ha_Hc
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.operators.op_converter import to_matrix_operator
import psutil
from qiskit.aqua.algorithms import QuantumAlgorithm, VQE
from operator_selector_new import _commutator
from itertools import chain
import time
import math
logger = logging.getLogger(__name__)


class ROTOADAPTVQE(IterativeVQE):
    """Create an instance of the ROTOADAPT-VQE algorithm.

    Parameters
    ----------
    operator_pool : OperatorPool
        Pool from which to draw new operators in the ansatz.
        See documentation for `qisresearch.adapt.operator_pool.OperatorPool` for construction.
    initial_state : Union[InitialState, None]
        Initial state of the register for the VQEs.
    vqe_optimizer : Optimizer
        See documentation for `qiskit.aqua.algorithm.VQE`
    hamiltonian : BaseOperator
        Operator to use for minimization.
    max_iters : int
        Maximum number of steps for ADAPT-VQE to take (new layers to add).
    energy_tol : float
        If the maximum energy change at any step is below this threshold, then the
        algorithm will terminate.
    max_evals_grouped : int
        See documentation for `qiskit.aqua.algorithm.VQE`
    aux_operators : List[Operator]
        See documentation for `qiskit.aqua.algorithm.VQE`
    auto_conversion : bool
        See documentation for `qiskit.aqua.algorithm.VQE`
    initial_parameters : Union[int, float]
        If `0`, then the initial parameter for each new layer is `0`. If it
        is '1' then use the optimized rotoparameter. If '2' then use
        the given 'float'.
    callback : callable
        See documentation for `qiskit.aqua.algorithm.VQE`
    step_callbacks : List[Callback]
        List of `Callback` objects to apply at each step.
    drop_duplicate_circuits : bool
        Whether or not to drop duplicate circuits at the gradient execution step.
        Possibly improves speed of gradient calculation step.
    return_best_result : bool
        Whether or not to return the best result for all the steps.
    parameter_tolerance : Union[None, float]
        If `float`, then circuits produced with parameters (absolute value) below
        this threshold will be ignored. This helps reduce the circuit depth if
        certain parameters are deemed not necessary in later steps in the algorithm.
        If `None` is passed, then this step is not done.
    compute_hessian : bool
        Whether or not to compute the Hessian at each layer of ROTOADAPT. The Hessian
        is defined by the expectation value of the double commutator `[[H, P], Q]`
        for operators `P` and `Q`.

    Attributes
    ----------
    commutators : List[Operator]
        The commutators of the Hamiltonian with each of the elements in the pool.
        Used for gradient evaluation.
    """
    #step history should be fine now? except we also want to keep track of number of evals necessary for computing next operator
    CONFIGURATION = {
        'name': 'ADAPTVQE',
        'description': 'ADAPT-VQE Algorithm',
    }

    def __init__(
            self,
            operator_pool: OperatorPool,
            initial_state: Union[InitialState, None],
            vqe_optimizer: Optimizer,
            hamiltonian: BaseOperator,
            max_iters: int = 10,
            energy_tol: float = 1e-3,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            initial_parameters: Union[int, float] = 0,
            callback=None,
            step_callbacks=[],
            drop_duplicate_circuits=True,
            return_best_result: bool = False,
            parameter_tolerance=None,
            compute_hessian: bool = False,
            operator_selector: OperatorSelector = None,
            parameters_per_step: int = 1,
            shortcut = False,
            reverse_opt = False,
            num_restarts = 1
    ):
        super().__init__(return_best_result)

        self.operator_pool = deepcopy(operator_pool)
        if initial_state is None:
            self.initial_state = Zero(num_qubits=operator_pool.num_qubits)
        else:
            self.initial_state = initial_state
        self.vqe_optimizer = vqe_optimizer
        self.hamiltonian = hamiltonian
        self.max_iters = max_iters
        self.energy_tol = energy_tol
        self.max_evals_grouped = max_evals_grouped
        self.aux_operators = aux_operators
        self.auto_conversion = auto_conversion
        self._compute_hessian = compute_hessian
        self._drop_duplicate_circuits = drop_duplicate_circuits
        self.callback = callback
        self.step_callbacks = step_callbacks
        self.ham_list = split_into_paulis(hamiltonian)
        self.shortcut = shortcut
        self.reverse_opt = reverse_opt
        self.num_restarts = num_restarts

        if operator_selector is None: #need to change, should be roto
            self._operator_selector = general_ROTOADAPT_OperatorSelector_pauli(
                self.hamiltonian,
                operator_pool=self.operator_pool,
                drop_duplicate_circuits=self._drop_duplicate_circuits,
                energy_tol = self.energy_tol
            )
        else:
            self._operator_selector = operator_selector

        if initial_parameters == 0:
            self.initial_parameters = 0
            self.__new_par = 0.0
        elif initial_parameters == 1:
            self.initial_parameters = 1
            self.__new_par = 0.0
        elif initial_parameters == 2:
            self.initial_parameters = 2
        elif initial_parameters == 2:
            self.initial_parameters = 2
        elif initial_parameters == 3:
            self.initial_parameters = 3
            self.__new_par = 0.0
        else:
            raise ValueError('Invalid option for new parameters supplied: {}'.format(initial_parameters))

        self.parameters_per_step = parameters_per_step
        self._parameter_tolerance = parameter_tolerance

        if len(self.step_callbacks) == 0: 
            self.step_callbacks.append(MinEnergyStopper(self.energy_tol))

        if self.reverse_opt:
            self.hamiltonian = -self.hamiltonian

    def _is_converged(self) -> bool:
        if self.step > self.max_iters:
            logger.info('Algorithm converged because max iterations ({}) reached'.format(self.max_iters))
            return True
        else:
            return False

    def first_vqe_kwargs(self) -> Dict:
        # This works for now, but always produces one extra parameter. -George, so we'll need to change this for rotoadapt too.
        id_op = WeightedPauliOperator.from_list(paulis=[Pauli.from_label('I' * self.operator_pool.num_qubits)],
                                                weights=[1.0])
        var_form = self.variational_form([id_op])

        self._operator_selector._quantum_instance = self.quantum_instance

        return {
            'operator': self.hamiltonian,
            'var_form': var_form,
            'optimizer': DummyOptimizer(),
            'initial_point': np.array([np.pi]),
            'max_evals_grouped': self.max_evals_grouped,
            'aux_operators': self.aux_operators,
            'callback': self.callback,
            'auto_conversion': self.auto_conversion
        }

    def next_vqe_kwargs(self, last_result) -> Dict:
        new_op_list = last_result['current_ops'] + [last_result['optimal op']]
        
        var_form = self.variational_form(new_op_list)
        if self.shortcut and self.initial_parameters == 3:
            self.__new_par = last_result['optimal param']
            print('MADE IT HERE 2')
        print('NEW PARAM', self._new_param)
        initial_point = np.concatenate((
            last_result['opt_params'],
            self._new_param
        ))
        if self.shortcut and len(self._step_history) < self.max_iters:
            optimizer = DummyOptimizer()
        else:
            optimizer = self.vqe_optimizer

        if self.reverse_opt and (len(self._step_history) == self.max_iters):
            self.hamiltonian = -self.hamiltonian
        return {
            'operator': self.hamiltonian,
            'var_form': var_form,
            'optimizer': optimizer,
            'initial_point': initial_point,
            'max_evals_grouped': self.max_evals_grouped,
            'aux_operators': self.aux_operators,
            'callback': self.callback,
            'auto_conversion': self.auto_conversion,
        }

    def post_process_result(self, result, vqe, last_result) -> Dict: 
        result = super().post_process_result(result, vqe, last_result)
        result['current_ops'] = deepcopy(vqe._var_form._operator_pool)
        print('result info:')
        print(result['energy'])
        print(result['opt_params'])
        if self._operator_selector.parameters_per_step == 2 and len(result['current_ops'])>1 and self._operator_selector.two_op_mode == False:
            intermed_circuit = self.variational_form(result['current_ops'][:-1]).construct_circuit(parameters = result['opt_params'][:-1])
            result['expec list 2'], evals = multi_circuit_eval(intermed_circuit, self.ham_list, qi = self._operator_selector.quantum_instance)
        else:
            intermed_circuit = None
        result['expec list'], evals = multi_circuit_eval(result['current_circuit'], self.ham_list, qi = self._operator_selector.quantum_instance) # evals, optimal_op, optimal_param, grad_info, Ha_info, Hc_info, energy_change_info, A_info, repeats
        result['num op choice evals'], result['optimal op'], result['optimal param'], result['grad info'], result['Ha info'], result['Hc info'], result['energy info'], result['A info'], result['repeats'], result['grads'], result['energy change'], result['Ha list'], result['A list'] = self._operator_selector.get_next_op_param(result, intermed_circuit)
        print("op selection info:")
        print('grad info', result['grad info'])
        print('energy info', result['energy info'])
        print('A info', result['A info'])
        print('Ha info', result['Ha info'])
        print('Hc info', result['Hc info'])
        print('repeats', result['repeats'])
        print('optimal op', result['optimal op'].print_details())
        if self._compute_hessian:
            hessian = self._operator_selector._hessian(circuit=result['current_circuit'])
        else:
            hessian = None
        result['hessian'] = hessian

        return result

    @property
    def _new_param(self):
        if self.initial_parameters == 2:
            output = [np.random.uniform(-np.pi, +np.pi) for i in range(self.parameters_per_step)]
        elif self.initial_parameters == 1:
            output = [np.pi/4 for i in range(self.parameters_per_step)]
        else:
            output = [self.__new_par for i in range(self.parameters_per_step)]
            print('MADE IT HERE')
        return np.array(output)

    def variational_form(self, ops):
        return ADAPTVariationalForm(
            operator_pool=ops,
            bounds=[(-np.pi, +np.pi)] * len(ops),
            initial_state=self.initial_state,
            tolerance=self._parameter_tolerance
        )
def get_H_term_energy(*indices, **kwargs):
    term_e_list = kwargs['he']
    indices = indices[0]
    H_term_energy = 0
    if indices:
        for i in indices:
            H_term_energy = H_term_energy + term_e_list[i][0]
    else:
        return 0
    return np.real(H_term_energy)

def get_energy_change_array(*terms, **kwargs):
    prev_energy = kwargs['energy']
    Hc = terms[0][0]
    Ha = terms[0][1]
    comm = terms[0][2][0]

    A = np.sqrt(Ha**2 + (comm**2)/4)
    energy = Hc - A
    return np.real(energy - prev_energy)

def get_energy_max_array(*terms, **kwargs):
    prev_energy = kwargs['energy']
    Hc = terms[0][0]
    Ha = terms[0][1]
    comm = terms[0][2][0]

    A = np.sqrt(Ha**2 + (comm**2)/4)
    energy = Hc + A
    return np.real(energy - prev_energy)

def get_A_array(*terms):
    Hc = terms[0][0]
    Ha = terms[0][1]
    comm = terms[0][2][0]

    A = np.sqrt(Ha**2 + (comm**2)/4)
    return np.real(A)


def convert_to_wpauli_list(term, *args):
    num_qubits = args[0]
    if term[0] == complex(0):
        separated_ham = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
    else:
        separated_ham = WeightedPauliOperator.from_list([term[1]],[term[0]])
    return separated_ham


def split_into_paulis(ham):
    args = [ham.num_qubits]
    ham_list = ham.paulis
    separated_ham_list = list(parallel_map(convert_to_wpauli_list, ham_list, args, num_processes = len(psutil.Process().cpu_affinity())))

    return separated_ham_list


def get_entry(list_entry, *args):
    num = args[0]
    return list_entry[num]


def check_com(op,**kwargs):
    flag = 0
    for op_2 in kwargs['ham_list']:
        if not op.commute_with(op_2):
            flag = 1
            break
    return flag

def get_sorted_list(op, **kwargs):
    term_list = kwargs['hp']
    energy_list = kwargs['he']
    Hcacb = 0
    Hcaab = 0
    Haacb = 0
    Haaab = 0

    for i,term in enumerate(term_list):
        if term.commute_with(op) and term.commute_with(kwargs['prev_op']):
            Hcacb = Hcacb + energy_list[i][0]
        elif term.commute_with(op) and not term.commute_with(kwargs['prev_op']):
            Hcaab = Hcaab + energy_list[i][0]
        elif not term.commute_with(op) and term.commute_with(kwargs['prev_op']):
            Haacb = Haacb + energy_list[i][0]
        else:
            Haaab = Haaab + energy_list[i][0]
    return [np.real(Hcacb), np.real(Hcaab), np.real(Haacb), np.real(Haaab)]


def truncate_decimal(num, decimal_digits):
    trunc_num = float(int(((10)**decimal_digits)*num))/(10**decimal_digits)
    return trunc_num


def multi_param_optimization(energy_info, **kwargs): #energy_info = list(zip(sort_list,HcomA_cb,HcomA_ab,Hca_comB,Haa_comB,double_comms))
    Hcacb = np.real(energy_info[0][0])
    Hcaab = np.real(energy_info[0][1])
    Haacb = np.real(energy_info[0][2])
    Haaab = np.real(energy_info[0][3])
    HcomA_cb = np.real(energy_info[1][0])
    HcomA_ab = np.real(energy_info[2][0])
    Hca_comB = np.real(energy_info[3][0])
    Haa_comB = np.real(energy_info[4][0])
    double_comm = np.real(energy_info[5][0])
    optimizer = kwargs['mini_optimizer']
    num_parameters = 2
    initial_point = np.array([1,1])
    bounds = np.array([(-np.pi, +np.pi)] * 2)
    def energy_eval(params):
        return Hcacb + np.cos(params[1])*Hcaab - 0.5*np.sin(params[1])*Hca_comB + np.cos(params[0])*(Haacb + np.cos(params[1])*Haaab - 0.5*np.sin(params[1])*Haa_comB) + 0.5*np.sin(params[0])*(HcomA_cb + np.cos(params[1])*HcomA_ab - 0.5*np.sin(params[1])*double_comm)
    opt_params, opt_val, num_optimizer_evals = optimizer.optimize(num_parameters, 
                                                              energy_eval,
                                                              variable_bounds=bounds,
                                                              initial_point=initial_point)
    return np.real(opt_val - kwargs['energy'])

def split_kwargs_Hc_term(op, ham_term_list):
    Hc = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    if op == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
        return 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    else:
        for term in ham_term_list:
            if term.commute_with(op):
                Hc = Hc + term
    return Hc

def split_kwargs_Ha_term(op, ham_term_list):
    Ha = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    if op == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
        return 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    else:
        for term in ham_term_list:
            if not term.commute_with(op):
                Ha = Ha + term
        return Ha



def split_op_Hc_term(ham, op):
    term_list = split_into_paulis(ham)
    Hc = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    for term in term_list:
        if term.commute_with(op):
            Hc = Hc + term
    return Hc

def split_op_Ha_term(ham, op):
    term_list = split_into_paulis(ham)
    Ha = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)])
    for term in term_list:
        if not term.commute_with(op):
            Ha = Ha + term
    return Ha



def get_sorted_list_squared(op_2,**kwargs):
    term_list = kwargs['hp']
    energy_list = kwargs['he']
    #[Hcacb, Hcaab, Haacb, Haaab]
    split_list = [[0,0,0,0] for i in range(len(kwargs['initial_op_list']))]
    k = 0
    for op_1 in kwargs['initial_op_list']:
        i = 0
        for i,term in enumerate(term_list):
            if term.commute_with(op_2) and term.commute_with(op_1):
                split_list[k][0] = split_list[k][0] + energy_list[i][0]
            elif term.commute_with(op_2) and not term.commute_with(op_1):
                split_list[k][1] = split_list[k][1] + energy_list[i][0]
            elif not term.commute_with(op_2) and term.commute_with(op_1):
                split_list[k][2] = split_list[k][2] + energy_list[i][0]
            else:
                split_list[k][3] = split_list[k][3] + energy_list[i][0]
        k = k + 1
    return split_list

def _commutator_squared(op,ham_list):
    comm_list = []
    for ham_term in ham_list:
        comm = _commutator(op,ham_term)
        if comm == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
            comm_list.append(5000000000*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]))
        else:
            comm_list.append(comm)

    return comm_list
def _double_commutator_squared(comm_op,**kwargs):
    num_qubits = kwargs['op_list'][0].num_qubits
    double_comm_list = []
    if comm_op == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)]) or comm_op == 5000000000*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)]):
        for op in kwargs['op_list']:
            double_comm_list.append(5000000000*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)]))
    else:
        for op in kwargs['op_list']:
            comm = _commutator(op,comm_op)
            if comm == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)]):
                double_comm_list.append(5000000000*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)]))
            else:
                double_comm_list.append(comm)
    return double_comm_list

def multi_param_optimization_squared(energy_info, **kwargs): #{'comm terms': self.comm_terms, 'comm energies': comm_energies_2D, 'op_list': self._operator_pool.pool. 'mini_optimizer': self.mini_optimizer, 'ham_list': self.ham_list}
    op_2 = energy_info[2]
    entry = energy_info[3]
    energy_array = []
    for i,op in enumerate(kwargs['op_list']):
        Hcacb = energy_info[0][i][0]
        Hcaab = energy_info[0][i][1]
        Haacb = energy_info[0][i][2]
        Haaab = energy_info[0][i][3]
        double_comm = energy_info[1][i][0]
        HcomA_cb = 0
        HcomA_ab = 0
        Hca_comB = 0
        Haa_comB = 0
        if double_comm == 5000000000:
            double_comm = 0
        for k,term in enumerate(kwargs['comm terms'][entry]):
            if kwargs['comm energies'][entry][k][0] != 5000000000:
                if term.commute_with(op):
                    HcomA_cb = HcomA_cb + kwargs['comm energies'][entry][k][0]
                else:
                    HcomA_ab = HcomA_ab + kwargs['comm energies'][entry][k][0]
            if kwargs['comm energies'][i][k][0] != 5000000000:
                if kwargs['ham_list'][k].commute_with(op_2):
                    Hca_comB = Hca_comB + kwargs['comm energies'][i][k][0]
                else:
                    Haa_comB = Haa_comB + kwargs['comm energies'][i][k][0]

        optimizer = kwargs['mini_optimizer']
        num_parameters = 2
        initial_point = np.array([1,1])
        bounds = np.array([(-np.pi, +np.pi)] * 2)
        def energy_eval(params):
            energy = Hcacb + np.cos(params[1])*Hcaab + 0.5*np.sin(params[1])*Hca_comB + np.cos(params[0])*(Haacb + np.cos(params[1])*Haaab + 0.5*np.sin(params[1])*Haa_comB) + 0.5*np.sin(params[0])*(HcomA_cb + np.cos(params[1])*HcomA_ab + 0.5*np.sin(params[1])*double_comm)
            return energy
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(num_parameters, 
                                                                  energy_eval,
                                                                  variable_bounds=bounds,
                                                                  initial_point=initial_point)
        energy_array.append(np.real(opt_val - kwargs['energy']))
    return energy_array





def order_ops(grads, energy_change_array, A_array, Hc_list, Ha_list, optimal_index):
    very_larg_num = 100000

    max_grad = 0
    min_grad = very_larg_num
    grad_num = 0
    grad_num_raw = 0
    grad_num_less = 0
    grad_num_less_raw = 0

    energy_min = very_larg_num
    energy_max = -very_larg_num
    energy_change_num = 0
    energy_change_num_raw = 0
    energy_change_num_less = 0
    energy_change_num_less_raw = 0

    A_max = 0
    A_min = very_larg_num
    A_num = 0
    A_num_raw = 0
    A_num_less = 0
    A_num_less_raw = 0

    Hc_min = very_larg_num
    Hc_max = -very_larg_num
    Hc_num = 0
    Hc_num_raw = 0
    Hc_num_less = 0
    Hc_num_less_raw = 0

    Ha_num = 0
    Ha_num_raw = 0
    Ha_max = 0
    Ha_min = very_larg_num
    Ha_num_less = 0
    Ha_num_less_raw = 0

    Hca_num = 0
    Hca_num_raw = 0
    Hca_max = -very_larg_num
    Hca_min = very_larg_num
    Hca_num_less = 0
    Hca_num_less_raw = 0


    decimal = 30

    for i in range(len(grads)):
        if truncate_decimal(abs(np.real(grads[i][0])), decimal) > truncate_decimal(abs(np.real(grads[optimal_index][0])), decimal):
            grad_num_raw = grad_num_raw + 1

        if truncate_decimal(abs(np.real(grads[i][0])), decimal) < truncate_decimal(abs(np.real(grads[optimal_index][0])), decimal):
            grad_num_less_raw = grad_num_less_raw + 1



        if truncate_decimal(energy_change_array[i],decimal) > truncate_decimal(energy_change_array[optimal_index],decimal):
            energy_change_num_raw = energy_change_num_raw + 1

        if truncate_decimal(energy_change_array[i],decimal) < truncate_decimal(energy_change_array[optimal_index],decimal):
            energy_change_num_less_raw = energy_change_num_less_raw + 1




        if truncate_decimal(abs(A_array[i]), decimal) > truncate_decimal(abs(A_array[optimal_index]),decimal):
            A_num_raw = A_num_raw + 1

        if truncate_decimal(abs(A_array[i]), decimal) < truncate_decimal(abs(A_array[optimal_index]),decimal):
            A_num_less_raw = A_num_less_raw + 1



        if truncate_decimal(Hc_list[i],decimal) < truncate_decimal(Hc_list[optimal_index],decimal):
            Hc_num_less_raw = Hc_num_less_raw + 1

        if truncate_decimal(Hc_list[i],decimal) > truncate_decimal(Hc_list[optimal_index],decimal):
            Hc_num_raw = Hc_num_raw + 1



        if truncate_decimal(abs(Ha_list[i]),decimal) > truncate_decimal(abs(Ha_list[optimal_index]), decimal):
            Ha_num_raw = Ha_num_raw + 1

        if truncate_decimal(abs(Ha_list[i]),decimal) < truncate_decimal(abs(Ha_list[optimal_index]), decimal):
            Ha_num_less_raw = Ha_num_less_raw + 1


        if truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) > truncate_decimal(Hc_list[optimal_index]-abs(Ha_list[optimal_index]), decimal):
            Hca_num_raw = Hca_num_raw + 1

        if truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) < truncate_decimal(Hc_list[optimal_index] - abs(Ha_list[optimal_index]), decimal):
            Hca_num_less_raw = Hca_num_less_raw + 1



        if truncate_decimal(abs(np.real(grads[i][0])),decimal) > truncate_decimal(abs(np.real(max_grad)),decimal) and truncate_decimal(abs(np.real(grads[i][0])),decimal) > truncate_decimal(abs(np.real(grads[optimal_index][0])),decimal):
            max_grad = grads[i][0]
            grad_num = grad_num + 1

        if truncate_decimal(abs(np.real(grads[i][0])),decimal) < truncate_decimal(abs(np.real(min_grad)),decimal) and truncate_decimal(abs(np.real(grads[i][0])),decimal) < truncate_decimal(abs(np.real(grads[optimal_index][0])),decimal):
            min_grad = grads[i][0]
            grad_num_less = grad_num_less + 1


        if truncate_decimal(energy_change_array[i],decimal) > truncate_decimal(energy_max,decimal) and truncate_decimal(energy_change_array[i],decimal) > truncate_decimal(energy_change_array[optimal_index],decimal):
            energy_max = energy_change_array[i]
            energy_change_num = energy_change_num + 1


        if truncate_decimal(energy_change_array[i],decimal) < truncate_decimal(energy_min,decimal) and truncate_decimal(energy_change_array[i],decimal) < truncate_decimal(energy_change_array[optimal_index],decimal):
            energy_min = energy_change_array[i]
            energy_change_num_less = energy_change_num_less + 1


        if truncate_decimal(abs(A_array[i]),decimal) > truncate_decimal(abs(A_max), decimal) and truncate_decimal(abs(A_array[i]),decimal) > truncate_decimal(abs(A_array[optimal_index]),decimal):
            A_max= A_array[i]
            A_num = A_num + 1

        if truncate_decimal(abs(A_array[i]),decimal) < truncate_decimal(abs(A_min), decimal) and truncate_decimal(abs(A_array[i]),decimal) < truncate_decimal(abs(A_array[optimal_index]),decimal):
            A_min= A_array[i]
            A_num_less = A_num_less + 1


        if truncate_decimal(Hc_list[i],decimal) > truncate_decimal(Hc_max,decimal) and truncate_decimal(Hc_list[i], decimal) > truncate_decimal(Hc_list[optimal_index],decimal):
            Hc_max = Hc_list[i]
            Hc_num = Hc_num + 1

        if truncate_decimal(Hc_list[i],decimal) < truncate_decimal(Hc_min,decimal) and truncate_decimal(Hc_list[i], decimal) < truncate_decimal(Hc_list[optimal_index],decimal):
            Hc_min = Hc_list[i]
            Hc_num_less = Hc_num_less + 1


        if truncate_decimal(abs(Ha_list[i]),decimal) > truncate_decimal(abs(Ha_max),decimal) and truncate_decimal(abs(Ha_list[i]),decimal) > truncate_decimal(abs(Ha_list[optimal_index]),decimal):
            Ha_max = Ha_list[i]
            Ha_num = Ha_num + 1

        if truncate_decimal(abs(Ha_list[i]),decimal) < truncate_decimal(abs(Ha_min),decimal) and truncate_decimal(abs(Ha_list[i]),decimal) < truncate_decimal(abs(Ha_list[optimal_index]),decimal):
            Ha_min = Ha_list[i]
            Ha_num_less = Ha_num_less + 1

        if truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) > truncate_decimal(Hca_max,decimal) and truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) > truncate_decimal(Hc_list[optimal_index] - abs(Ha_list[optimal_index]),decimal):
            Hca_max = Hc_list[i] - abs(Ha_list[i])
            Hca_num = Hca_num + 1

        if truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) < truncate_decimal(Hca_min,decimal) and truncate_decimal(Hc_list[i] - abs(Ha_list[i]),decimal) < truncate_decimal(Hc_list[optimal_index] - abs(Ha_list[optimal_index]),decimal):
            Hca_min = Hc_list[i] - abs(Ha_list[i])
            Hca_num_less = Hca_num_less + 1



    if max_grad == 0:
        max_grad = grads[optimal_index][0]

    if min_grad == very_larg_num:
        min_grad = grads[optimal_index][0]

    if energy_max == -very_larg_num:
        energy_max = energy_change_array[optimal_index]

    if energy_min == very_larg_num:
        energy_min = energy_change_array[optimal_index]

    if A_max == 0:
        A_max = A_array[optimal_index]

    if A_min == very_larg_num:
        A_min = A_array[optimal_index]

    if Hc_max == -very_larg_num:
        Hc_max = Hc_list[optimal_index]

    if Hc_min == very_larg_num:
        Hc_min = Hc_list[optimal_index]

    if Ha_max == 0:
        Ha_max = Ha_list[optimal_index]

    if Ha_min == very_larg_num:
        Ha_min = Ha_list[optimal_index]

    if Hca_max == -very_larg_num:
        Hca_max = Hc_list[optimal_index] - abs(Ha_list[optimal_index])

    if Hca_min == very_larg_num:
        Hca_min = Hc_list[optimal_index] - abs(Ha_list[optimal_index])

    print('Hca info', [Hca_num_less_raw, Hca_num_less, Hca_min, Hc_list[optimal_index] - abs(Ha_list[optimal_index]), Hca_max, Hca_num, Hca_num_raw])

    grad_info = [grad_num_less_raw, grad_num_less, min_grad, grads[optimal_index][0], max_grad, grad_num, grad_num_raw]
    energy_change_info = [energy_change_num_less_raw, energy_change_num_less, energy_min, energy_change_array[optimal_index], energy_max, energy_change_num, energy_change_num_raw]
    A_info =  [A_num_less_raw, A_num_less, A_min, A_array[optimal_index], A_max, A_num, A_num_raw]
    Hc_info = [Hc_num_less_raw, Hc_num_less, Hc_min, Hc_list[optimal_index], Hc_max, Hc_num, Hc_num_raw]
    Ha_info = [Ha_num_less_raw, Ha_num_less, Ha_min, Ha_list[optimal_index], Ha_max, Ha_num, Ha_num_raw]

    return grad_info, energy_change_info, A_info, Ha_info, Hc_info


def get_Ha_indices(op, ham_list):
    Ha_indices = []
    for i,term in enumerate(ham_list):
        if not term.commute_with(op):
            Ha_indices.append(i)
    return tuple(Ha_indices)

def get_Hc_indices(op, ham_list):
    Hc_indices = []
    for i,term in enumerate(ham_list):
        if term.commute_with(op):
            Hc_indices.append(i)
    return tuple(Hc_indices)

def gt_sumop_flag(info):
    if np.sign(np.real(info[0][0])) == np.sign(np.real(info[1])):
        return 0

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

def get_unique_comms_and_indices(op_pool, ham_list):
    unique_comms = []
    meta_index_list = []
    num_qubits = op_pool[0].num_qubits
    big_list = [0]*4**num_qubits
    for i,op in enumerate(op_pool):
        index_list = []
        for term in ham_list:
            comm = _commutator(op,term)
            if comm != 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
                comm_info = comm.paulis
                comm_weight = comm_info[0][0]
                #if check_if_in_ham(comm,ham_list):
                #    index_list.append([-flag,comm_weight])
                #else:
                comm_pauli = comm_info[0][1]
                comm_pauli = WeightedPauliOperator.from_list([comm_pauli],[1])
                comm_name = comm_pauli.print_details()[:num_qubits]
                comm_entry = base_4_map(comm_name)
                if unique_comms:
                    if big_list[int(comm_entry)]:
                        index_list.append([big_list[int(comm_entry)] - 1, comm_weight])
                    else:
                        big_list[int(comm_entry)] = len(unique_comms) + 1
                        index_list.append([len(unique_comms),comm_weight])
                        unique_comms.append(comm_pauli)
                else:
                    big_list[int(comm_entry)] = 1
                    unique_comms.append(comm_pauli)
                    index_list.append([0,comm_weight])
            else:
                index_list.append([0,0])
        meta_index_list.append(index_list)
    return tuple(meta_index_list), unique_comms

def check_if_in_ham(comm,ham_list):
    flag = 0
    for i,term in enumerate(ham_list):
        if comm == term:
            flag = i + 1
            break
    return flag


def get_grads(*comm_indices, expecs, result_expecs, ham_weight_list):
    comm_indices = comm_indices[0]
    grad = 0
    if comm_indices:
        for index_pair in comm_indices:
            if index_pair[0] >= 0:
                grad = grad + index_pair[1]*expecs[index_pair[0]][0]
            else:
                grad = grad + index_pair[1]*result_expecs[-(index_pair[0] - 1)]*(1/ham_weight_list[-(index_pair[0] - 1)])
        return tuple([grad, 0])
    else:
        return tuple([0,0])

def get_2_p_grads(*comm_indices, expecs, mode):
    grad_list = []
    comm_indices = comm_indices[0]
    if mode == 'comA':
        if comm_indices:
            for index_set in comm_indices:
                if index_set:
                    grad = 0
                    for index_pair in index_set:
                        grad = grad + index_pair[1]*expecs[index_pair[0]][0]
            return tuple([grad, 0])
        else:
            return tuple([0,0])
    else:
        return 0

def get_approx_energy(*terms, **kwargs):
    prev_energy = kwargs['energy']
    Hc = terms[0][0]
    Ha = terms[0][1]
    approx_energy = Hc - abs(Ha)
    return np.real(approx_energy - prev_energy)
#careful, organization is different between the Hxa_comB's and the HcomA_xb's
def get_unique_2_param_comms_and_indices(op_pool, ham_list):
        unique_measurements = []
        num_qubits = op_pool[0].num_qubits
        meta_meta_Hca_comB_index_list =[]
        meta_meta_Haa_comB_index_list =[]
        meta_meta_HcomA_cb_index_list =[]
        meta_meta_HcomA_ab_index_list =[]
        meta_meta_double_index_list = []
        big_list = [0]*4**num_qubits
        for op in op_pool:
            meta_Hca_comB_index_list =[]
            meta_Haa_comB_index_list =[]
            meta_HcomA_cb_index_list =[]
            meta_HcomA_ab_index_list =[]
            meta_double_index_list = []
            for op_2 in op_pool:
                Hca_comB_index_list =[]
                Haa_comB_index_list =[]
                HcomA_cb_index_list =[]
                HcomA_ab_index_list =[]
                double_index_list = []
                for term in ham_list:
                    comm = _commutator(op,term)
                    if comm != 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
                        comm_info = comm.paulis
                        comm_weight = comm_info[0][0]
                        comm_pauli = comm_info[0][1]
                        comm_pauli = WeightedPauliOperator.from_list([comm_pauli],[1])
                        comm_name = comm_pauli.print_details()[:num_qubits]
                        comm_entry = base_4_map(comm_name)
                        if unique_measurements:
                            if big_list[int(comm_entry)]:
                                if comm.commute_with(op_2):
                                    HcomA_cb_index_list.append([big_list[int(comm_entry)] - 1, comm_weight])
                                else:
                                    HcomA_ab_index_list.append([big_list[int(comm_entry)] - 1, comm_weight])
                                if term.commute_with(op_2):
                                    Hca_comB_index_list.append([big_list[int(comm_entry)] - 1, comm_weight])
                                else:
                                    Haa_comB_index_list.append([big_list[int(comm_entry)] - 1, comm_weight])
                            else:
                                big_list[int(comm_entry)] = len(unique_comms) + 1
                                if comm.commute_with(op_2):
                                    HcomA_cb_index_list.append([len(unique_measurements), comm_weight])
                                else:
                                    HcomA_ab_index_list.append([len(unique_measurements), comm_weight])
                                if term.commute_with(op_2):
                                    Hca_comB_index_list.append([len(unique_measurements), comm_weight])
                                else:
                                    Haa_comB_index_list.append([len(unique_measurements), comm_weight])
                                unique_measurements.append(comm_pauli)
                        else:
                            big_list[int(comm_entry)] = 1
                            unique_measurements.append(comm_pauli)
                            if comm.commute_with(op_2):
                                HcomA_cb_index_list.append([0, comm_weight])
                            else:
                                HcomA_ab_index_list.append([0, comm_weight])
                            if term.commute_with(op_2):
                                Hca_comB_index_list.append([0, comm_weight])
                            else:
                                Haa_comB_index_list.append([0, comm_weight])
                        comm_2 = _commutator(op_2,term)
                        if comm_2 != 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*op.num_qubits)]):
                            comm_2_info = comm_2.paulis
                            comm_2_weight = comm_2_info[0][0]
                            comm_2_pauli = comm_2_info[0][1]
                            comm_2_pauli = WeightedPauliOperator.from_list([comm_2_pauli],[1])
                            comm_2_name = comm_2_pauli.print_details()[:num_qubits]
                            comm_2_entry = base_4_map(comm_2_name)
                            if unique_measurements:
                                if big_list[int(comm_2_entry)]:
                                    double_index_list.append([big_list[int(comm_2_entry)] - 1, comm_2_weight])
                                else:
                                    big_list[int(comm_2_entry)] = len(unique_measurements) + 1
                                    double_index_list.append([len(unique_comms),comm_2_weight])
                                    unique_measurements.append(comm_2_pauli)
                            else:
                                big_list[int(comm_2_entry)] = 1
                                unique_measurements.append(comm_2_pauli)
                                double_index_list.append([0,comm_2_weight])
                        else:
                            double_index_list.append([0,0])
                    else:
                        Hca_comB_index_list.append([0,0])
                        Haa_comB_index_list.append([0,0])
                        HcomA_cb_index_list.append([0,0])
                        HcomA_ab_index_list.append([0,0])
                        double_index_list.append([0,0])
                meta_Hca_comB_index_list.append(Hca_comB_index_list)
                meta_Haa_comB_index_list.append(Haa_comB_index_list)
                meta_HcomA_cb_index_list.append(HcomA_cb_index_list)
                meta_HcomA_ab_index_list.append(HcomA_ab_index_list)
                meta_double_index_list.append(double_index_list)
            meta_meta_Hca_comB_index_list.append(meta_Hca_comB_index_list)
            meta_meta_Haa_comB_index_list.append(meta_Haa_comB_index_list)
            meta_meta_HcomA_cb_index_list.append(meta_HcomA_cb_index_list)
            meta_meta_HcomA_ab_index_list.append(meta_HcomA_ab_index_list)
            meta_meta_double_index_list.append(meta_double_index_list)
        return tuple(meta_meta_Hca_comB_index_list), tuple(meta_meta_Haa_comB_index_list), tuple(meta_meta_HcomA_cb_index_list), tuple(meta_meta_HcomA_ab_index_list), tuple(meta_meta_double_index_list), unique_measurements

class general_ROTOADAPT_OperatorSelector_pauli(OperatorSelector, VQE):
    def __init__(
            self, 
            hamiltonian, 
            operator_pool: OperatorPool, 
            drop_duplicate_circuits: bool = True, 
            parameters_per_step: int = 1,
            energy_tol = 1e-3,
            op_mode = "energy",
            param_mode = 1,
            reverse = False,
            tacking = False,
            mini_optimizer = None,
            two_op_mode = False,
            num_processes = len(psutil.Process().cpu_affinity()),
            split_sets= True,
            sum_ops = False,
            approx_best_op = False,
            fastboi = True,
            startup = False):
        super().__init__(hamiltonian, operator_pool, drop_duplicate_circuits, None)
        self.parameters_per_step = parameters_per_step
        self.hamiltonian = hamiltonian
        self.ham_list = split_into_paulis(hamiltonian)
        self.ham_weight_list = self.hamiltonian.paulis[0]
        self.op_mode = op_mode
        self.param_mode = param_mode
        self.reverse  = reverse
        self.tacking = tacking
        self.iter = 0
        self.prev_op_index = 0
        self.two_op_mode = two_op_mode
        self.split_sets = split_sets
        self.sum_ops = sum_ops
        self.approx_best_op = approx_best_op
        self.startup = startup
        print('reducing pool:')
        kwargs = {'ham_list': self.ham_list}
        self.num_processes = num_processes
        self.num_qubits = hamiltonian.num_qubits
        start = time.time()
        print('starting pool reduction')
        flag_list = list(parallel_map(check_com, self._operator_pool.pool, task_kwargs = kwargs, num_processes = self.num_processes))
        new_op_list = []
        for i,op in enumerate(self._operator_pool.pool):
            if flag_list[i]:
                new_op_list.append(op.print_details()[:self.num_qubits])
        self._operator_pool = PauliPool.from_pauli_strings(new_op_list)
        stop_time = time.time()
        red_time = stop_time - start
        print('pool reduced to:', len(self._operator_pool.pool), 'in', red_time)
        start = time.time()
        self.Ha_indices = list(parallel_map(get_Ha_indices,self._operator_pool.pool, task_kwargs = kwargs, num_processes = self.num_processes))
        self.Hc_indices = list(parallel_map(get_Hc_indices,self._operator_pool.pool, task_kwargs = kwargs, num_processes = self.num_processes))
        self.comm_index_list, self.unique_comm_terms = get_unique_comms_and_indices(self._operator_pool.pool, self.ham_list)
        print('normal comms len', len(self._operator_pool.pool)*len(self.ham_list))
        print('pared down comms len', len(self.unique_comm_terms))
        print('indices acquired', time.time()-start)
        if parameters_per_step == 2:
            self.mini_optimizer = mini_optimizer
        #    self.Hca_comB_indices, self.Haa_comB_indices, self.HcomA_cb_indices, self.HcomA_ab_indices, self.double_comms, self.unique_2_comms = get_unique_2_param_comms_and_indices(self._operator_pool.pool, self.ham_list)
        if two_op_mode == True:
            self.mini_optimizer = mini_optimizer
            self.comm_terms = list(parallel_map(_commutator_squared,self._operator_pool.pool, task_kwargs = {'ham_list': self.ham_list}, num_processes =self.num_processes))
            self.comms_total = list(parallel_map(_commutator, self._operator_pool.pool, task_kwargs = {'hamiltonian':self.hamiltonian},num_processes =self.num_processes))
            self.double_comms = list(parallel_map(_double_commutator_squared, self.comms_total, task_kwargs = {'op_list':self._operator_pool.pool}, num_processes =self.num_processes))
            self.flattened_double_comms = list(chain.from_iterable(self.double_comms))
            self.flattened_comm_terms = list(chain.from_iterable(self.comm_terms))
        self.already_done = False


    def get_next_op_param(self, result, intermed_circuit = None):
        """
            method: get_next_op_param
            args:
                result- the data for the recently calculated result
            returns:
                comma separated info about the next op choice
        """
        if (self.parameters_per_step == 1 and self.approx_best_op == False) or (self.parameters_per_step == 2 and len(result['current_ops']) < 2 and self.two_op_mode == False):#normally would be 2 but we have extra op
            if self.tacking:
                if (self.iter % 2) == 0:
                    self.op_mode = "energy"
                else:
                    self.op_mode = "adapt"
                self.iter = self.iter + 1
            if self.startup:
                if self.iter > 2:
                    self.op_mode = "Ha max"
                else:
                    self.op_mode = "energy"
                self.iter = self.iter + 1
            start = time.time()
            kwargs = {'he': result['expec list']}
            Ha_list = list(parallel_map(get_H_term_energy, self.Ha_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            stop = time.time() - start
            print("got Ha list", stop)
            kwargs = {'indices': self.Hc_indices, 'he': result['expec list']}
            Hc_list = list(parallel_map(get_H_term_energy, self.Hc_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            stop = time.time() - start
            print("got Hc list", stop)
            start = time.time()
            intermed = start
            if self.split_sets:
                grads_t = []
                evals_t = 0
                range_len = math.floor(len(self.unique_comm_terms)/self.num_processes)
                for i in range(self.num_processes + 1):
                    print('started loop')
                    stopper = (i+1)*range_len
                    starter = i*range_len
                    if stopper > len(self.unique_comm_terms):
                        stopper = len(self.unique_comm_terms)
                    if starter == len(self.unique_comm_terms):
                        break
                    print(starter)
                    print(stopper)
                    grads, evals = multi_circuit_eval(
                    result['current_circuit'], 
                    self.unique_comm_terms[starter:stopper], 
                    qi=self.quantum_instance, 
                    drop_dups=self._drop_duplicate_circuits
                    )
                    grads_t = grads_t+grads
                    evals_t = evals_t + evals
                    stop = time.time() - intermed
                    print("got grads", stop)
                    intermed = stop
                grads = grads_t
                evals = evals_t
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads, 'result_expecs': result['expec list'], 'ham_weight_list': self.ham_weight_list}, num_processes = self.num_processes))

            else:
                grads, evals = multi_circuit_eval(
                                result['current_circuit'], 
                                self.unique_comm_terms, 
                                qi=self.quantum_instance, 
                                drop_dups=self._drop_duplicate_circuits
                                )
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads, 'result_expecs': result['expec list'], 'ham_weight_list': self.ham_weight_list}, num_processes = self.num_processes))
            stop = time.time() - start
            print('total grad eval time:', stop)
            ziplist = list(zip(Hc_list, Ha_list, grads))
            kwargs = {'energy': result['energy']}
            energy_change_array = list(parallel_map(get_energy_change_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            energy_max_list = list(parallel_map(get_energy_max_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            A_array = list(parallel_map(get_A_array, ziplist, num_processes = self.num_processes))
            approx_energy_array = list(parallel_map(get_approx_energy,ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            args =tuple([0])
            grad_list = list(parallel_map(get_entry, grads, args, num_processes = self.num_processes))
            grad_list = list(parallel_map(abs, grad_list, num_processes = self.num_processes))

            if self.op_mode == "adapt":
                if self.sum_ops:
                    args =tuple([0])
                    grad_list = list(parallel_map(get_entry, grads, args, num_processes = self.num_processes))
                    sum_flag_list = list(parallel_map(get_sumop_flag, zip(grads, Ha_list), num_processes = self.num_processes))
                else:
                    args =tuple([0])
                    grad_list = list(parallel_map(get_entry, grads, args, num_processes = self.num_processes))
                    grad_list = list(parallel_map(abs, grad_list, num_processes = self.num_processes))
                    optimal_index_tuple = np.where(np.array(grad_list) == np.array(grad_list).max())

            elif self.op_mode == "energy":
                if self.sum_ops:
                    do_this
                    #not yet supported
                else:
                    optimal_index_tuple = np.where(np.array(energy_change_array) == np.array(energy_change_array).min())

            elif self.op_mode == "A max":
                A_list = list(parallel_map(abs, A_array, num_processes = self.num_processes))
                optimal_index_tuple = np.where(np.array(A_list) == np.array(A_list).max())

            elif self.op_mode == "Hc min":
                optimal_index_tuple = np.where(np.array(Hc_list) == np.array(Hc_list).min())

            elif self.op_mode == "energy max":
                optimal_index_tuple = np.where(np.array(energy_max_list) == np.array(energy_max_list).max())
            elif self.op_mode == "Ha max":
                args =tuple([0])
                optimal_index_tuple = np.where(np.array(Ha_list) == np.array(Ha_list).max())

            else:
                args =tuple([0])
                #Ha_abs_list = list(parallel_map(abs, Ha_list, num_processes = self.num_processes))
                optimal_index_tuple = np.where(np.array(Ha_list) == np.array(Ha_list).max())
                grad_max = 0
                grad_list = list(parallel_map(get_entry, grads, args, num_processes = self.num_processes))
                grad_list = list(parallel_map(abs, grad_list, num_processes = self.num_processes))
                for i,index in enumerate(optimal_index_tuple[0]):
                    if grad_list[index] > grad_max:
                        grad_max = grad_list[index]
                        meta_op_ind= i

            meta_op_ind = 0

            optimal_index_list = optimal_index_tuple[0]

            repeats = len(optimal_index_list)

            optimal_index = optimal_index_list[meta_op_ind]

            grad_info, energy_change_info, A_info, Ha_info, Hc_info = order_ops(grads, energy_change_array, A_array, Hc_list, Ha_list, optimal_index)

            optimal_op = self._operator_pool.pool[optimal_index]

            optimal_param = (np.arctan2(np.real(grads[optimal_index][0]),2*np.real(Ha_list[optimal_index])) - np.pi)/2
            print('optimal param', optimal_param)

            return evals, optimal_op, optimal_param, grad_info, Ha_info, Hc_info, energy_change_info, A_info, repeats, grad_list, energy_change_array, Ha_list, A_array

        if self.parameters_per_step == 2 and len(result['current_ops']) > 1 and self.two_op_mode == False: 
            self.intermed_circuit = intermed_circuit
            kwargs = {'hp': self.ham_list, 'he': result['expec list 2'], 'prev_op': result['current_ops'][-1]}
            sort_list = list(parallel_map(get_sorted_list, self._operator_pool.pool, task_kwargs = kwargs, num_processes =self.num_processes))


            Hca_terms = list(parallel_map(split_kwargs_Hc_term,self._operator_pool.pool,task_kwargs = {'ham_term_list': self.ham_list}, num_processes = self.num_processes))
            Haa_terms = list(parallel_map(split_kwargs_Ha_term,self._operator_pool.pool,task_kwargs = {'ham_term_list': self.ham_list}, num_processes = self.num_processes))
            Hca_comB_terms = list(parallel_map(_commutator,Hca_terms,task_kwargs = {'hamiltonian': result['current_ops'][-1]}, num_processes = self.num_processes))
            Haa_comB_terms = list(parallel_map(_commutator,Haa_terms,task_kwargs = {'hamiltonian': result['current_ops'][-1]}, num_processes = self.num_processes))


            Hca_comB_empty_index_list = []
            for i,entry in enumerate(Hca_comB_terms):
                if entry == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)]):
                    Hca_comB_empty_index_list.append(i)
                    Hca_comB_terms.remove(0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)]))
            if len(Hca_comB_terms) == 0:
                Hca_comB = [[0,0] for i in inrange(len(self._operator_pool.pool))]
                evals1 = 0
            else:
                Hca_comB, evals1 = multi_circuit_eval(self.intermed_circuit, Hca_comB_terms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)

            if len(Hca_comB_empty_index_list) > 0:
                Hca_comB_empty_index_list.reverse()
                for index in Hca_comB_empty_index_list:
                    Hca_comB.insert(index,[0,0])

            Haa_comB_empty_index_list = []
            for i,entry in enumerate(Haa_comB_terms):
                if entry == 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)]):
                    Haa_comB_empty_index_list.append(i)
                    Haa_comB_terms.remove(0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*self.num_qubits)]))
            if len(Haa_comB_terms) == 0:
                Haa_comB = [[0,0] for i in range(len(self._operator_pool.pool))]
                evals2 = 0
            else:
                Haa_comB, evals2 = multi_circuit_eval(self.intermed_circuit, Haa_comB_terms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)

            if len(Haa_comB_empty_index_list) > 0:
                Haa_comB_empty_index_list.reverse()
                for index in Haa_comB_empty_index_list:
                    Haa_comB.insert(index,[0,0])


            HcomA_terms = list(parallel_map(_commutator,self._operator_pool.pool, task_kwargs = {'hamiltonian': self.hamiltonian}, num_processes = self.num_processes))
            HcomA_cb_terms = list(parallel_map(split_op_Hc_term, HcomA_terms, task_kwargs = {'op': result['current_ops'][-1]}, num_processes = self.num_processes))
            HcomA_ab_terms = list(parallel_map(split_op_Ha_term, HcomA_terms, task_kwargs = {'op': result['current_ops'][-1]}, num_processes = self.num_processes))
            HcomA_cb, evals3 = multi_circuit_eval(self.intermed_circuit, HcomA_cb_terms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)
            HcomA_ab, evals4 = multi_circuit_eval(self.intermed_circuit, HcomA_ab_terms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)


            double_comm_terms = list(parallel_map(_commutator,HcomA_terms,task_kwargs = {'hamiltonian': result['current_ops'][-1]}, num_processes = self.num_processes))

            double_comms, evals5 = multi_circuit_eval(
                self.intermed_circuit, 
                double_comm_terms, 
                qi=self.quantum_instance, 
                drop_dups=self._drop_duplicate_circuits
                )
            evals = evals1 + evals2 + evals3 + evals4 + evals5
            energy_info = list(zip(sort_list,HcomA_cb,HcomA_ab,Hca_comB,Haa_comB,double_comms,self._operator_pool.pool))
            kwargs = {'mini_optimizer': self.mini_optimizer, 'energy': result['energy']}
            energies_2_param = list(parallel_map(multi_param_optimization, energy_info, task_kwargs = kwargs, num_processes = self.num_processes))

            optimal_index_tuple = np.where(np.array(energies_2_param) == np.array(energies_2_param).min())
            optimal_index_list = optimal_index_tuple[0]
            repeats = len(optimal_index_list)
            optimal_index = optimal_index_list[0]




            #checking against others:
            start = time.time()
            kwargs = {'he': result['expec list']}
            Ha_list = list(parallel_map(get_H_term_energy, self.Ha_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            kwargs = {'indices': self.Hc_indices, 'he': result['expec list']}
            Hc_list = list(parallel_map(get_H_term_energy, self.Hc_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            stop = time.time() - start
            start = time.time()
            if self.split_sets:
                grads_t = []
                evals_t = 0
                range_len = math.floor(len(self.unique_comm_terms)/self.num_processes)
                for i in range(self.num_processes + 1):
                    stopper = (i+1)*range_len
                    starter = i*range_len
                    if stopper > len(self.unique_comm_terms):
                        stopper = len(self.unique_comm_terms)
                    if starter == len(self.unique_comm_terms):
                        break
                    grads, evals = multi_circuit_eval(
                    result['current_circuit'], 
                    self.unique_comm_terms[starter:stopper], 
                    qi=self.quantum_instance, 
                    drop_dups=self._drop_duplicate_circuits
                    )
                    grads_t = grads_t+grads
                    evals_t = evals_t + evals
                grads = grads_t
                evals = evals_t
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))

            else:
                grads, evals = multi_circuit_eval(
                                result['current_circuit'], 
                                self.unique_comm_terms, 
                                qi=self.quantum_instance, 
                                drop_dups=self._drop_duplicate_circuits
                                )
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))
            stop = time.time() - start
            print('grad eval time:', stop)
            ziplist = list(zip(Hc_list, Ha_list, grads))
            kwargs = {'energy': result['energy']}
            energy_change_array = list(parallel_map(get_energy_change_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            energy_max_list = list(parallel_map(get_energy_max_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            A_array = list(parallel_map(get_A_array, ziplist, num_processes = self.num_processes))

            optimal_energy_1_index = np.argmin(energy_change_array)

            grad_info, energy_change_info, A_info, Ha_info, Hc_info = order_ops(grads, energy_change_array, A_array, Hc_list, Ha_list, optimal_index)
            print('energy_1_param', energy_change_array[optimal_energy_1_index])
            print('energy_2_param', energies_2_param[optimal_index])


            optimal_op = self._operator_pool.pool[optimal_index]

            if self.param_mode == 1:
                optimal_param = -np.arctan2(np.real(Ha_list[optimal_index]),2*np.real(grads[optimal_index][0]))
            elif self.param_mode == 2:
                optimal_param = np.random.uniform(0,2*np.pi)
            else: 
                optimal_param = 0

            return evals, optimal_op, optimal_param, grad_info, Ha_info, Hc_info, energy_change_info, A_info, repeats

        if self.parameters_per_step == 2 and self.two_op_mode == True and self.already_done == False:
            kwargs = {'hp': self.ham_list, 'he': result['expec list'], 'initial_op_list': self._operator_pool.pool}
            sort_list = list(parallel_map(get_sorted_list_squared, self._operator_pool.pool, task_kwargs = kwargs, num_processes =self.num_processes))
            set_length = len(self.ham_list)
            sets = len(self._operator_pool.pool)
            comm_energies, evals1 = multi_circuit_eval(result['current_circuit'], self.flattened_comm_terms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)
            comm_energies_2D  = []
            for i in range(sets):
                comm_energies_2D.append(comm_energies[i*set_length:(i+1)*set_length])
            double_comm_energies, evals2 = multi_circuit_eval(result['current_circuit'], self.flattened_double_comms, qi=self.quantum_instance, drop_dups = self._drop_duplicate_circuits)
            double_comms_2D = []
            for i in range(sets):
                double_comms_2D.append(double_comm_energies[i*sets:(i+1)*sets])
            energy_info = list(zip(sort_list, double_comms_2D, self._operator_pool.pool, range(len(self._operator_pool.pool))))
            energies_2_param = list(parallel_map(multi_param_optimization_squared, energy_info, task_kwargs = {'comm terms': self.comm_terms, 'comm energies': comm_energies_2D, 'op_list': self._operator_pool.pool, 'mini_optimizer': self.mini_optimizer, 'ham_list': self.ham_list, 'energy': result['energy']}, num_processes = self.num_processes))
            flattened_energies_2_param = list(chain.from_iterable(energies_2_param))
            optimal_index_tuple = np.where(np.array(flattened_energies_2_param) == np.array(flattened_energies_2_param).min())
            optimal_index_list = optimal_index_tuple[0]
            repeats = len(optimal_index_list)
            optimal_index = optimal_index_list[0]
            index = optimal_index
            print('energies_2_param', flattened_energies_2_param[optimal_index])
            counter = 0
            while index >= 0:
                counter = counter + 1
                index = index - sets
            counter = counter - 1
            index = index + sets
            op_1_index = index
            op_2_index = counter
            self.op_2_index = op_2_index
            evals = evals1 + evals2

            optimal_index = op_1_index

            start = time.time()
            kwargs = {'he': result['expec list']}
            Ha_list = list(parallel_map(get_H_term_energy, self.Ha_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            kwargs = {'indices': self.Hc_indices, 'he': result['expec list']}
            Hc_list = list(parallel_map(get_H_term_energy, self.Hc_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            stop = time.time() - start
            start = time.time()
            if self.split_sets:
                grads_t = []
                evals_t = 0
                range_len = math.floor(len(self.unique_comm_terms)/self.num_processes)
                for i in range(self.num_processes + 1):
                    stopper = (i+1)*range_len
                    starter = i*range_len
                    if stopper > len(self.unique_comm_terms):
                        stopper = len(self.unique_comm_terms)
                    if starter == len(self.unique_comm_terms):
                        break
                    grads, evals = multi_circuit_eval(
                    result['current_circuit'], 
                    self.unique_comm_terms[starter:stopper], 
                    qi=self.quantum_instance, 
                    drop_dups=self._drop_duplicate_circuits
                    )
                    grads_t = grads_t+grads
                    evals_t = evals_t + evals
                grads = grads_t
                evals = evals_t
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))

            else:
                grads, evals = multi_circuit_eval(
                                result['current_circuit'], 
                                self.unique_comm_terms, 
                                qi=self.quantum_instance, 
                                drop_dups=self._drop_duplicate_circuits
                                )
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))
            stop = time.time() - start
            print('grad eval time:', stop)
            ziplist = list(zip(Hc_list, Ha_list, grads))
            kwargs = {'energy': result['energy']}
            energy_change_array = list(parallel_map(get_energy_change_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            energy_max_list = list(parallel_map(get_energy_max_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            A_array = list(parallel_map(get_A_array, ziplist, num_processes = self.num_processes))

            optimal_energy_1_index = np.argmin(energy_change_array)

            grad_info, energy_change_info, A_info, Ha_info, Hc_info = order_ops(grads, energy_change_array, A_array, Hc_list, Ha_list, optimal_index)


            optimal_op = self._operator_pool.pool[optimal_index]

            if self.param_mode == 1:
                optimal_param = -np.arctan2(np.real(Ha_list[optimal_index]),2*np.real(grads[optimal_index][0]))
            elif self.param_mode == 2:
                optimal_param = np.random.uniform(0,2*np.pi)
            else: 
                optimal_param = 0
            self.already_done = True
            self.prev_repeats = repeats
            return evals, optimal_op, optimal_param, grad_info, Ha_info, Hc_info, energy_change_info, A_info, repeats, grads, energy_change_array, Ha_list, A_array

        if self.already_done == True:
            optimal_op = self._operator_pool.pool[self.op_2_index]
            optimal_index = self.op_2_index
            repeats = self.prev_repeats
            self.already_done = False


            start = time.time()
            kwargs = {'he': result['expec list']}
            Ha_list = list(parallel_map(get_H_term_energy, self.Ha_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            kwargs = {'indices': self.Hc_indices, 'he': result['expec list']}
            Hc_list = list(parallel_map(get_H_term_energy, self.Hc_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            stop = time.time() - start
            start = time.time()
            if self.split_sets:
                grads_t = []
                evals_t = 0
                range_len = math.floor(len(self.unique_comm_terms)/self.num_processes)
                for i in range(self.num_processes + 1):
                    stopper = (i+1)*range_len
                    starter = i*range_len
                    if stopper > len(self.unique_comm_terms):
                        stopper = len(self.unique_comm_terms)
                    if starter == len(self.unique_comm_terms):
                        break
                    grads, evals = multi_circuit_eval(
                    result['current_circuit'], 
                    self.unique_comm_terms[starter:stopper], 
                    qi=self.quantum_instance, 
                    drop_dups=self._drop_duplicate_circuits
                    )
                    grads_t = grads_t+grads
                    evals_t = evals_t + evals
                grads = grads_t
                evals = evals_t
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))

            else:
                grads, evals = multi_circuit_eval(
                                result['current_circuit'], 
                                self.unique_comm_terms, 
                                qi=self.quantum_instance, 
                                drop_dups=self._drop_duplicate_circuits
                                )
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))
            stop = time.time() - start
            print('grad eval time:', stop)
            ziplist = list(zip(Hc_list, Ha_list, grads))
            kwargs = {'energy': result['energy']}
            energy_change_array = list(parallel_map(get_energy_change_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            energy_max_list = list(parallel_map(get_energy_max_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            A_array = list(parallel_map(get_A_array, ziplist, num_processes = self.num_processes))

            optimal_energy_1_index = np.argmin(energy_change_array)

            grad_info, energy_change_info, A_info, Ha_info, Hc_info = order_ops(grads, energy_change_array, A_array, Hc_list, Ha_list, optimal_index)

            optimal_op = self._operator_pool.pool[optimal_index]

            if self.param_mode == 1:
                optimal_param = -np.arctan2(np.real(Ha_list[optimal_index]),2*np.real(grads[optimal_index][0]))
            elif self.param_mode == 2:
                optimal_param = np.random.uniform(0,2*np.pi)
            else: 
                optimal_param = 0

            return evals, optimal_op, optimal_param, grad_info, Ha_info, Hc_info, energy_change_info, A_info, repeats


        if self.parameters_per_step == 1 and self.approx_best_op == True:

            start = time.time()
            kwargs = {'he': result['expec list']}
            Ha_list = list(parallel_map(get_H_term_energy, self.Ha_indices, task_kwargs = kwargs, num_processes = self.num_processes))
            kwargs = {'indices': self.Hc_indices, 'he': result['expec list']}
            Hc_list = list(parallel_map(get_H_term_energy, self.Hc_indices, task_kwargs = kwargs, num_processes = self.num_processes))


            stop = time.time() - start
            start = time.time()
            if self.split_sets:
                grads_t = []
                evals_t = 0
                range_len = math.floor(len(self.unique_comm_terms)/self.num_processes)
                for i in range(self.num_processes + 1):
                    stopper = (i+1)*range_len
                    starter = i*range_len
                    if stopper > len(self.unique_comm_terms):
                        stopper = len(self.unique_comm_terms)
                    if starter == len(self.unique_comm_terms):
                        break
                    grads, evals = multi_circuit_eval(
                    result['current_circuit'], 
                    self.unique_comm_terms[starter:stopper], 
                    qi=self.quantum_instance, 
                    drop_dups=self._drop_duplicate_circuits
                    )
                    grads_t = grads_t+grads
                    evals_t = evals_t + evals
                grads = grads_t
                evals = evals_t
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))

            else:
                grads, evals = multi_circuit_eval(
                                result['current_circuit'], 
                                self.unique_comm_terms, 
                                qi=self.quantum_instance, 
                                drop_dups=self._drop_duplicate_circuits
                                )
                grads = list(parallel_map(get_grads, self.comm_index_list, task_kwargs = {'expecs': grads}, num_processes = self.num_processes))
            stop = time.time() - start
            print('grad eval time:', stop)
            ziplist = list(zip(Hc_list, Ha_list, grads))
            kwargs = {'energy': result['energy']}
            energy_change_array = list(parallel_map(get_energy_change_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            energy_max_list = list(parallel_map(get_energy_max_array, ziplist, task_kwargs = kwargs, num_processes = self.num_processes))
            A_array = list(parallel_map(get_A_array, ziplist, num_processes = self.num_processes))





class MinEnergyStopper(Callback):
    def __init__(self, min_energy_tolerance: float):
        self.min_energy_tolerance = min_energy_tolerance

    def halt(self, step_history) -> bool:
        min_energy = step_history[-1]['energy']
        if (len(step_history) > 1):
            second_min_energy = step_history[-2]['energy']
        else:
            second_min_energy = 100000000
        return abs(second_min_energy - min_energy) < self.min_energy_tolerance
        
    def halt_reason(self, step_history):
        return 'Energy threshold satisfied'











