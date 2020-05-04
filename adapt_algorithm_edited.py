import copy
import time #added by willB or WillB
import logging
from typing import List, Tuple, Union

import networkx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.initial_states import InitialState, Zero, Custom
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
from qiskit.tools.parallel import parallel_map

from .adapt_variational_form import ADAPTVariationalForm, MixerLayer, CompositeVariationalForm
from .operator_pool import OperatorPool, PauliPool

logger = logging.getLogger(__name__)


def _circ_eval(op, **kwargs):
    return op.construct_evaluation_circuit(**kwargs)


def _hash(circ):
    return hash(str(circ))


def _compute_grad(op, **kwargs):
    return op.evaluate_with_result(**kwargs)


def _commutator(op, hamiltonian=None, gradient_tolerance=None):
    return 1j * (hamiltonian * op - op * hamiltonian).chop(threshold=gradient_tolerance, copy=True)

def fast_circ_eq(circ_1, circ_2):
    """Quickly determines if two circuits are equal.
    Not for general use, this is not equivalent to `circ_1 == circ_2`.
    This function simply compares the data, since this is sufficient for
    dropping duplicate circuits in ADAPTVQE.

    Parameters
    ----------
    circ_1 : QuantumCircuit
        First circuit to compare.
    circ_2 : QuantumCircuit
        Second circuit to compare.

    Returns
    -------
    bool
        Whether or not the circuits are equal.

    """
    if len(circ_1._data) != len(circ_2._data):
        return False
    data_1 = reversed(circ_1._data)
    data_2 = reversed(circ_2._data)
    for d_1, d_2 in zip(data_1, data_2):
        # i = (op, qubits, other)
        op_1, q_1, ot_1 = d_1
        op_2, q_2, ot_2 = d_2
        if q_1 != q_2:
            return False
        if ot_1 != ot_2:
            return False
        if op_1 != op_2:
            return False
    return True


def fast_circuit_inclusion(circ, circ_list):
    """Quickly determines whether a circuit is included in a list.
    Not for general use, see `fast_circ_eq`.

    Parameters
    ----------
    circ : QuantumCircuit
        Circuit to check inclusion of.
    circ_list : List[QuantumCircuit]
        List where `circ` might be.

    Returns
    -------
    bool
        Whether or not the circuit is in the list, based on `fast_circ_eq`.

    """
    for c in circ_list:
        if fast_circ_eq(c, circ):
            return True
    return False


class ADAPTVQE(QuantumAlgorithm):
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
            grad_tol: float = 1e-3,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            use_zero_initial_parameters=False
    ):
        super().__init__()
        self.operator_pool = copy.deepcopy(operator_pool)
        if initial_state is None:
            self.initial_state = Zero(num_qubits=operator_pool.num_qubits)
        else:
            self.initial_state = initial_state
        self.vqe_optimizer = vqe_optimizer
        self.hamiltonian = hamiltonian
        self.max_iters = max_iters
        self.grad_tol = grad_tol
        self.max_evals_grouped = max_evals_grouped
        self.aux_operators = aux_operators
        self.auto_conversion = auto_conversion
        self.use_zero_initial_parameters = use_zero_initial_parameters
        self.parameters_per_step = 1

        self._coms = None
        self.first_step = False
        self._current_max_grad = None
        self._optimal_circuit = None
        self._ret = {}

        self.adapt_step_history = {
            'gradient_list': [],  # updated in _compute_gradients
            'max_gradient': [],  # updated in _ansatz_operator_list
            'optimal_parameters': [],  # updated in _run
            'circuit': [],  # updated in _run
            'operators': [],  # updated in _ansatz_operator_list
            'vqe_ret': [],  # updated in _run
            'energy_history': [],  # updated in _run
            'Total Eval Time': 0, #added by WillB
            'Total num evals': 0  #added by WillB
        }

    @property
    def _new_param(self):
        if self.use_zero_initial_parameters:
            if self.first_step:
                return [0.0]
            return [0.0 for i in range(self.parameters_per_step)]
        else:
            if self.first_step:
                return [np.random.uniform(-np.pi, +np.pi)]
            return [np.random.uniform(-np.pi, +np.pi) for i in range(self.parameters_per_step)]

    def _run(self) -> dict:
        start_time = time.time() #Added by WillB
        logger.info('Starting ADAPT step {} of maximum {}'.format(1, self.max_iters))
        circuit = self.initial_state.construct_circuit(mode='circuit')
        params = self._new_param
        op_list = self._ansatz_operator_list(current_circuit=circuit, current_ops=[])
        vqe = self._vqe_run(operator_list=op_list, initial_parameters=params)
        self.adapt_step_history['optimal_parameters'].append(vqe['optimal_params'])
        self.adapt_step_history['circuit'].append(vqe['optimal_circuit'])
        self.adapt_step_history['vqe_ret'].append(vqe['_ret'])
        self.adapt_step_history['Total num evals'] += vqe['_ret']['num_optimizer_evals'] #added by WillB
        self.adapt_step_history['energy_history'].append(vqe['_ret']['energy'])
        logger.info('Finished ADAPT step {} of maximum {} with energy {}'.format(1, self.max_iters, vqe['_ret']['energy']))
        iters = 1
        print(iters)
        while iters <= self.max_iters: #and self._current_max_grad > self.grad_tol: #removed by WillB
            logger.info('Starting ADAPT step {} of maximum {}'.format(iters, self.max_iters))
            circuit = vqe['optimal_circuit']
            params = np.concatenate((vqe['optimal_params'], self._new_param))
            op_list = self._ansatz_operator_list(current_circuit=circuit, current_ops=op_list)
            #if self._current_max_grad < self.grad_tol: #removed by willB
            #    break
            vqe = self._vqe_run(operator_list=op_list, initial_parameters=params)
            self.adapt_step_history['optimal_parameters'].append(vqe['optimal_params'])
            self.adapt_step_history['circuit'].append(vqe['optimal_circuit'])
            self.adapt_step_history['vqe_ret'].append(vqe['_ret'])
            self.adapt_step_history['Total num evals'] += vqe['_ret']['num_optimizer_evals'] #added by WillB
            self.adapt_step_history['energy_history'].append(vqe['_ret']['energy'])
            logger.info(
                'Finished ADAPT step {} of maximum {} with energy {}'.format(iters, self.max_iters,
                                                                             vqe['_ret']['energy']))
            iters += 1
            print(iters)
        logger.info('Finished final ADAPT step {} of maximum {}, with final energy {}'.format(
            iters,
            self.max_iters,
            vqe['_ret']['energy']
        ))
        logger.info('Final gradient is {} where tolerance is {}'.format(
            self._current_max_grad, self.grad_tol
        ))

        self._optimal_circuit = circuit
        self._ret = vqe['_ret']
        del self.adapt_step_history['vqe_ret'] #added by WillB
        stop_time = time.time() - start_time #added by WillB
        self.adapt_step_history['Total Eval Time'] = stop_time #added by WillB
        return self.adapt_step_history

    def _compute_gradients(self, circuit: QuantumCircuit) -> List[Tuple[complex, complex]]:
        kwargs = {'statevector_mode': self.quantum_instance.is_statevector}
        logger.info('Constructing evaluation circuits...')
        total_evaluation_circuits = list(parallel_map(
            _circ_eval,
            self.commutators,
            task_kwargs={**kwargs, 'wave_function': circuit},
            num_processes=aqua_globals.num_processes
        ))
        total_evaluation_circuits = [item for sublist in total_evaluation_circuits for item in sublist]
        logger.info('Removing duplicate circuits')
        final_circs = []
        for circ in total_evaluation_circuits:
            if not fast_circuit_inclusion(circ, final_circs):
                final_circs.append(circ)
        logger.info('Finished removing duplicate circuits')
        logger.debug('Executing {} circuits for gradient evaluation...'.format(len(final_circs)))
        result = self.quantum_instance.execute(final_circs)
        logger.debug('Computing {} gradients...'.format(len(self.commutators)))
        grads = list(parallel_map(
            _compute_grad,
            self.commutators,
            task_kwargs={**kwargs, 'result': result},
            num_processes=aqua_globals.num_processes
        ))
        self.adapt_step_history['Total num evals'] += len(final_circs)
        logger.debug('Computed gradients: {}'.format(grads))
        return [abs(tup[0]) for tup in grads]

    @property
    def commutators(self) -> List[BaseOperator]:
        if self._coms is not None:
            return self._coms
        logger.info('Computing commutators...')
        self._coms = list(parallel_map(
            _commutator,
            self.operator_pool.pool,
            task_kwargs={'hamiltonian': self.hamiltonian, 'gradient_tolerance': self.grad_tol},
            num_processes=aqua_globals.num_processes
        ))  # type: List[BaseOperator]
        logger.info('Computed {} commutators'.format(len(self._coms)))
        if all(isinstance(op, WeightedPauliOperator) for op in self._coms):
            new_coms = []
            new_pool = []
            for com, op in zip(self._coms, self.operator_pool.pool):
                if len(com.paulis) > 0:
                    new_coms.append(com)
                    new_pool.append(op)
            self._coms = new_coms
            self.operator_pool._pool = new_pool
            logger.info('Dropped commuting terms, new pool has size {}'.format(len(self._coms)))
        else:
            logger.info(
                'Dropping commuting terms currently only supported for WeightedPauliOperator class')
        if len(self._coms) == 0:
            raise ValueError('List of commutators is empty.')
        return self._coms

    def _ansatz_operator_list(self, current_circuit: QuantumCircuit,
                              current_ops: List[BaseOperator]) -> List[BaseOperator]:
        grads = self._compute_gradients(circuit=current_circuit)
        self._current_max_grad = np.max(grads)
        self.adapt_step_history['max_gradient'].append(self._current_max_grad)
        new_op = self.operator_pool.pool[np.argmax(grads)]  # type: BaseOperator
        logger.info('New operator with gradient {} added to ansatz: {}'.format(
            self._current_max_grad,
            str(new_op)
        ))
        self.adapt_step_history['operators'].append(new_op.print_details())
        return current_ops + [new_op]

    def _vqe_run(self, operator_list: List[BaseOperator], initial_parameters: np.ndarray,
                 **kwargs) -> dict:

        self._current_operator_list = operator_list
        self._current_initial_parameters = initial_parameters

        var_form = self.variational_form()

        vqe = VQE(
            operator=self.hamiltonian,
            var_form=var_form,
            optimizer=self.vqe_optimizer,
            initial_point=initial_parameters,
            max_evals_grouped=self.max_evals_grouped,
            aux_operators=self.aux_operators,
            callback=None,
            auto_conversion=self.auto_conversion
        )
        vqe_result = vqe.run(self.quantum_instance, **kwargs)  # == vqe._ret

        return {
            'optimal_circuit': vqe.get_optimal_circuit(),
            'optimal_params': vqe.optimal_params,
            '_ret': vqe_result
        }

    def variational_form(self):
        return ADAPTVariationalForm(
            operator_pool=self._current_operator_list,
            bounds=[(-np.pi, +np.pi)] * len(self._current_operator_list),
            initial_state=self.initial_state
        )

    def get_optimal_circuit(self):
        if self._optimal_circuit is None:
            raise AquaError(
                "Cannot find optimal circuit before running the algorithm to find optimal params.")
        return self._optimal_circuit


class ADAPTQAOA(ADAPTVQE):
    CONFIGURATION = {
        'name': 'ADAPTQAOA',
        'description': 'ADAPT-QAOA Algorithm',
    }

    def __init__(
            self,
            operator_pool: OperatorPool,
            initial_state: Union[InitialState, None],
            vqe_optimizer: Optimizer,
            hamiltonian: BaseOperator,
            max_iters: int = 10,
            grad_tol: float = 1e-8,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            use_zero_initial_parameters=False
    ):
        super().__init__(
            operator_pool=operator_pool,
            initial_state=initial_state,
            vqe_optimizer=vqe_optimizer,
            hamiltonian=hamiltonian,
            max_iters=max_iters,
            grad_tol=grad_tol,
            max_evals_grouped=max_evals_grouped,
            aux_operators=aux_operators,
            auto_conversion=auto_conversion,
            use_zero_initial_parameters=use_zero_initial_parameters
        )

        self.parameters_per_step = 2
        self.first_step = True

        if initial_state is None:
            self.initial_state = Custom(hamiltonian.num_qubits, state='uniform')
        else:
            self.initial_state = initial_state

    def _ansatz_operator_list(self, current_circuit: QuantumCircuit,
                              current_ops: List[BaseOperator]) -> List[BaseOperator]:
        if self.first_step:
            self._current_max_grad = self.grad_tol + 1
            self.adapt_step_history['operators'].extend([self.hamiltonian])
            self.first_step = False
            return current_ops + [self.hamiltonian]
        grads = self._compute_gradients(circuit=current_circuit)
        self._current_max_grad = np.max(grads)
        self.adapt_step_history['max_gradient'].append(self._current_max_grad)
        new_op = self.operator_pool.pool[np.argmax(grads)]  # type: BaseOperator
        logger.info('New operator with gradient {} added to ansatz: {}'.format(
            self._current_max_grad,
            str(new_op)
        ))
        self.adapt_step_history['operators'].extend([new_op.print_details, self.hamiltonian])
        return current_ops + [new_op, self.hamiltonian]


class GraphADAPTQAOA(ADAPTVQE):

    def __init__(
            self,
            initial_state: Union[InitialState, None],
            vqe_optimizer: Optimizer,
            graph: np.array,
            hamiltonian,
            pauli_pairs,
            max_iters: int = 10,
            grad_tol: float = 1e-3,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            use_zero_initial_parameters=False
    ):

        self.graph = graph
        g = networkx.Graph(self.graph)
        operator_pool = PauliPool.from_coupling_list_pairs(
            coupling_list=[list(e) for e in g.edges],
            pauli_pairs=pauli_pairs
        )

        super().__init__(
            operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters,
            grad_tol,
            max_evals_grouped,
            aux_operators,
            auto_conversion,
            use_zero_initial_parameters
        )
        self.parameters_per_step = 2

    def variational_form(self) -> VariationalForm:
        var_form_list = ()
        for op in self._current_operator_list:
            vf_cost = ADAPTVariationalForm([op], bounds=[(-np.pi, +np.pi)], initial_state=None)
            vf_mix = MixerLayer(op.num_qubits, mixer_str='X'*op.num_qubits)
            var_form_list += (vf_cost, vf_mix)  # type Tuple[VariationalForm]

        vf = CompositeVariationalForm(var_form_list, initial_state=self.initial_state)
        return vf

