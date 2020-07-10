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
from operator_selector_new import OperatorSelector, multi_circuit_eval, get_Ha
from qiskit.aqua import aqua_globals, QuantumInstance
import psutil

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
            parameters_per_step: int = 1
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

        if operator_selector is None: #need to change, should be roto
            self._operator_selector = ROTOADAPTOperatorSelector_pauli(
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
        else:
            raise ValueError('Invalid option for new parameters supplied: {}'.format(initial_parameters))

        self.parameters_per_step = parameters_per_step
        self._parameter_tolerance = parameter_tolerance

        if len(self.step_callbacks) == 0: 
            self.step_callbacks.append(MinEnergyStopper(self.energy_tol))

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
        print(last_result['optimal op'].print_details())

        if self.initial_parameters == 1:
            self.__new_param = last_result['optimal param']

        del last_result['optimal op']
        del last_result['optimal param']
        
        var_form = self.variational_form(new_op_list)
        initial_point = np.concatenate((
            last_result['opt_params'],
            self._new_param
        ))
        return {
            'operator': self.hamiltonian,
            'var_form': var_form,
            'optimizer': self.vqe_optimizer,
            'initial_point': initial_point,
            'max_evals_grouped': self.max_evals_grouped,
            'aux_operators': self.aux_operators,
            'callback': self.callback,
            'auto_conversion': self.auto_conversion,
        }

    def post_process_result(self, result, vqe, last_result) -> Dict: 
        result = super().post_process_result(result, vqe, last_result)
        result['current_ops'] = deepcopy(vqe._var_form._operator_pool)
        result['expec list'], evals = multi_circuit_eval(result['current_circuit'], self.ham_list, qi = self._operator_selector.quantum_instance)
        result['num op choice evals'], result['optimal op'], result['optimal param'] = self._operator_selector.get_next_op_param(result)
        print(result['energy'])
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
        else:
            output = [self.__new_par for i in range(self.parameters_per_step)]
        return np.array(output)

    def variational_form(self, ops):
        return ADAPTVariationalForm(
            operator_pool=ops,
            bounds=[(-np.pi, +np.pi)] * len(ops),
            initial_state=self.initial_state,
            tolerance=self._parameter_tolerance
        )


class ROTOADAPTOperatorSelector_pauli(OperatorSelector):
    def __init__(
            self, 
            hamiltonian, 
            operator_pool: OperatorPool, 
            drop_duplicate_circuits: bool = True, 
            energy_tol: float = None,
            parameters_per_step: int = 1):
        super().__init__(hamiltonian, operator_pool, drop_duplicate_circuits, None)
        self.parameters_per_step = parameters_per_step
        self.energy_tol = energy_tol
        self.ham_list = split_into_paulis(hamiltonian)
    def get_next_op_param(self, result):
        """
        	method: get_energy_param_lists
        	args:
        		result- the data for the recently calculated result
        	returns:
        		dict with number of energy evaluations required, array of optimized energies for each operator in pool,
        		 array of optimized parameters for the energy values in energy array
        """
        if self.parameters_per_step == 1:
            args = tuple()
            kwargs = {'hp': self.ham_list, 'he': result['expec list']}
            Ha_list = list(parallel_map(get_Ha, self._operator_pool.pool, task_kwargs = kwargs, num_processes = len(psutil.Process().cpu_affinity())))
            Hc_list = list(parallel_map(get_Hc, self._operator_pool.pool, task_kwargs = kwargs, num_processes = len(psutil.Process().cpu_affinity())))
            grads, evals = multi_circuit_eval(
                            result['current_circuit'], 
                            self.commutators, 
                            qi=self.quantum_instance, 
                            drop_dups=self._drop_duplicate_circuits
                            )
            #ziplist = list(zip(Hc_list, Ha_list, grads))
            ziplist = list(zip(Hc_list, Ha_list, grads))
            energy_array = list(parallel_map(get_optimal_array, ziplist, num_processes = len(psutil.Process().cpu_affinity())))
            optimal_energy_index = np.argmin(energy_array)
            #optimal_energy_index = np.where(np.array(energy_array) == np.array(energy_array).min())
            #optimal_energy_index = optimal_energy_index[0]
            #print(optimal_energy_index)
            #if len(optimal_energy_index) > 1:
            #    entry = np.random.randint(0,len(optimal_energy_index) - 1)
            #    optimal_energy_index = optimal_energy_index[entry]
            #else:
            #    optimal_energy_index = optimal_energy_index[0]
            optimal_op = self._operator_pool.pool[optimal_energy_index]
            optimal_param = -np.arctan2(np.real(Ha_list[optimal_energy_index]),2*np.real(grads[optimal_energy_index][0]))
            return evals, optimal_op, optimal_param


def get_optimal_array(*terms):
    Hc = terms[0][0]
    Ha = terms[0][1]
    comm = terms[0][2][0]

    A = np.sqrt(Ha**2 + (comm**2)/4)
    Energy = Hc - A
    return Energy


def get_Hc(op, **kwargs):
    term_list = kwargs['hp']
    term_e_list = kwargs['he']
    Hc_energy = 0
    for i,term in enumerate(term_list):
        if term.commute_with(op):
            Hc_energy = Hc_energy + term_e_list[i][0]
    return Hc_energy

def convert_to_wpauli_list(term, *args):
    num_qubits = args[0]
    if term[0] == complex(0):
        separated_ham = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
    else:
        separated_ham = WeightedPauliOperator.from_list([term[1]],[term[0]])
    return separated_ham


def split_into_paulis(ham):
    """
        Used to split the given hamiltonian into lists of constituent parts:
        weight_list: list of complex floats that are the weights of terms in the hamiltonian
        pauli_list: a list of the Weighted_pauli_operator representations of the pauli strings in the hamiltonian
                    (does not include weights)
        name_list: a list of strings that correspond to the names of the operators in the hamiltonian
    """
    args = [ham.num_qubits]
    ham_list = ham.paulis
    separated_ham_list = parallel_map(convert_to_wpauli_list, ham_list, args,num_processes = len(psutil.Process().cpu_affinity()))

    return separated_ham_list



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


def find_commutator(op_2, kwargs):
    op_1 = kwargs['op_1']
    return op_1*op_2 - op_2*op_1
def does_commute(op_2, kwargs):
    op_1 = kwargs['op_1']
    return op_2.commute_with(op_1)



