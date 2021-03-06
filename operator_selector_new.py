from abc import abstractmethod
from itertools import product
from typing import List, Tuple
import logging
import numpy as np

from qiskit import QuantumCircuit
from qisresearch.adapt import OperatorPool
from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
from qiskit.tools import parallel_map
from qiskit.aqua import aqua_globals, QuantumInstance
import psutil
from qisresearch.adapt.operator_pool import OperatorPool, PauliPool, CompletePauliPool
from copy import deepcopy
import time

logger = logging.getLogger(__name__)

def _circ_eval(op, **kwargs):
    return op.construct_evaluation_circuit(**kwargs)

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

def _compute_grad(op, **kwargs):
    return op.evaluate_with_result(**kwargs) #will added


def _commutator(op, hamiltonian=None, gradient_tolerance=None):
    return 1j * (op * hamiltonian - hamiltonian * op).chop(threshold=gradient_tolerance, copy=True)


def multi_circuit_eval(circuit: QuantumCircuit, op_list: List[BaseOperator], qi: QuantumInstance, drop_dups: bool = True):
    num_processes = 8
    kwargs = {'statevector_mode': qi.is_statevector}
    logger.info('Constructing evaluation circuits...')
    start = time.time()
    print('about to get eval circs')
    total_evaluation_circuits = list(parallel_map(
        _circ_eval,
        op_list,
        task_kwargs={**kwargs, 'wave_function': circuit},
        num_processes=num_processes)
    )
    total_evaluation_circuits = [item for sublist in total_evaluation_circuits for item in sublist]
    logger.info('Removing duplicate circuits')
    start = time.time()
    print('about to drop dups')
    if drop_dups:
        final_circs = []
        for circ in total_evaluation_circuits:
            if not fast_circuit_inclusion(circ, final_circs):
                final_circs.append(circ)
        logger.info('Finished removing duplicate circuits')
    else:
        final_circs = deepcopy(total_evaluation_circuits)
    del total_evaluation_circuits
    evals = len(final_circs) #will added
    logger.debug('Executing {} circuits for evaluation...'.format(len(final_circs)))
    start = time.time()
    #print('about to execute multi circuit evals')
    result = qi.execute(final_circs)
    #print('finished executing multi circuit evals')
    logger.debug('Computing {} expectations...'.format(len(op_list)))
    start = time.time()
    print('about to compute expecs')
    exp_vals = list(parallel_map(
        _compute_grad,
        op_list,
        task_kwargs={**kwargs, 'result': result},
        num_processes=num_processes)
    )
    logger.debug('Computed expectations: {}'.format(exp_vals))
    return exp_vals, evals #will added


class OperatorSelector:

    def __init__(self, hamiltonian, operator_pool: OperatorPool, drop_duplicate_circuits: bool = True, grad_tol: float = None):
        self._hamiltonian = hamiltonian
        self._operator_pool = operator_pool
        self._coms = None
        self._dcoms = None
        self._quantum_instance = None
        self._drop_duplicate_circuits = drop_duplicate_circuits
        self._grad_tol = grad_tol
    
    @property
    def quantum_instance(self):
        if self._quantum_instance is None:
            raise ValueError('Quantum instance has not been set yet')
        else:
            return self._quantum_instance
    
    @abstractmethod
    def get_new_operator_list(self, grad_list, current_ops) -> List[BaseOperator]:
        pass
    
    @property
    def commutators(self) -> List[BaseOperator]:
        """Returns the commutators of the Hamiltonian with each of the elements
        in the pool. Commutators that are `0` are dropped. If all the elements are
        dropped, an error is raised. Duplicate operators are also removed.

        Returns
        -------
        List[BaseOperator]
            Description of returned object.

        """
        if self._coms is not None:
            return self._coms
        logger.info('Computing commutators...')
        self._coms = list(parallel_map(
            _commutator,
            self._operator_pool.pool,
            task_kwargs={'hamiltonian': self._hamiltonian},
            num_processes=aqua_globals.num_processes)
        ) # type: List[BaseOperator]
        logger.info('Computed {} commutators'.format(len(self._coms)))
        if all(isinstance(op, WeightedPauliOperator) for op in self._coms):
            new_coms = []
            new_pool = []
            for com, op in zip(self._coms, self._operator_pool.pool):
                if len(com.paulis) > 0:
                    new_coms.append(com)
                    new_pool.append(op)
            self._coms = new_coms
            self._operator_pool._pool = new_pool
            logger.info('Dropped commuting terms, new pool has size {}'.format(len(self._coms)))
        else:
            logger.info(
                'Dropping commuting terms currently only supported for WeightedPauliOperator class')
        if len(self._coms) == 0:
            raise ValueError('List of commutators is empty.')
        return self._coms
    
    @property
    def double_commutators(self):
        if self._dcoms is not None:
            return self._dcoms
        n_ops = len(self._operator_pool.pool)
        array_out = np.empty((n_ops, n_ops), dtype=WeightedPauliOperator)
        def comp_com(a, b):
            return a*b - b*a
        for i, j in product(range(n_ops), repeat=2):
            op_1 = self._operator_pool.pool[i]
            op_2 = self._operator_pool.pool[j]
            array_out[i, j] = comp_com(comp_com(self._hamiltonian, op_1), op_2)
        self._dcoms = array_out
        return self._dcoms

    def _compute_gradients(self, circuit: QuantumCircuit) -> List[Tuple[complex, complex]]:
        grads, evals = multi_circuit_eval(
            circuit, 
            self.commutators, 
            qi=self.quantum_instance, 
            drop_dups=self._drop_duplicate_circuits
            )
        logger.debug('Computed gradients: {}'.format(grads))
        grad_vals,std_vals = list(zip(*grads))
        grad_vals = np.abs(np.array(grad_vals))
        max_index = np.argmax(grad_vals)
        max_grad = grads[max_index]
        return grads,max_grad,evals
    
    def _hessian(self, circuit: QuantumCircuit):
        kwargs = {'statevector_mode': self.quantum_instance.is_statevector}
        logger.info('Constructing evaluation circuits for Hessian...')
        dcoms = self.double_commutators.flatten().tolist()
        total_evaluation_circuits = list(parallel_map(
            _circ_eval,
            dcoms,
            task_kwargs={**kwargs, 'wave_function': circuit},
            num_processes=len(psutil.Process().cpu_affinity())
        ))
        total_evaluation_circuits = [item for sublist in total_evaluation_circuits for item in sublist]
        logger.info('Removing duplicate circuits')
        if self._drop_duplicate_circuits:
            final_circs = []
            for circ in total_evaluation_circuits:
                if not fast_circuit_inclusion(circ, final_circs):
                    final_circs.append(circ)
            logger.info('Finished removing duplicate circuits')
        else:
            final_circs = total_evaluation_circuits
        logger.debug('Executing {} circuits for Hessian evaluation...'.format(len(final_circs)))
        result = self.quantum_instance.execute(final_circs)
        logger.debug('Computing Hessian...')
        
        n_ops = len(self._operator_pool.pool)
        hess = np.zeros((n_ops, n_ops))
        for i, j in product(range(n_ops), repeat=2):
            hess[i, j], _ = self.double_commutators[i,j].evaluate_with_result(
                result=result,
                **kwargs
            )
        logger.debug('Computed Hessian')
        return hess


class ADAPTOperatorSelector(OperatorSelector):

    def get_new_operator_list(self, grad_list: List[Tuple[complex, complex]], current_ops, circuit: QuantumCircuit = None) -> Tuple[List[BaseOperator], float]:
        grads, stds = list(zip(*grad_list))
        grads = np.abs(np.array(grads))
        max_index = np.argmax(grads)
        #max_index = np.where(np.array(grads) == np.array(grads).max()) #will added
        #max_index = max_index[0]
        #print(max_index)
        #if len(max_index) > 1:
        #    entry = np.random.randint(0,len(max_index) - 1)
        #    max_index = max_index[entry]
        #else:
        #    max_index = max_index[0]
        new_op = self._operator_pool.pool[max_index]
        max_grad = grad_list[max_index]
        return current_ops + [new_op]


class AntiCommutingSelector(OperatorSelector):

    def get_new_operator_list(self, grad_list: List[Tuple[complex, complex]], current_ops, circuit: QuantumCircuit = None) -> Tuple[List[BaseOperator], float]:
        grads, stds = list(zip(*grad_list))
        grads = np.abs(np.array(grads))
        max_index = np.argmax(grads)
        new_op = self._operator_pool.pool[max_index]
        max_grad = grad_list[max_index]
        if abs(max_grad[0]) >= self._grad_tol:
            return current_ops + [new_op]
        else:
            anti_exp_vals = self.compute_anti_exp_vals(circuit)
            exp_val_list = [d['mean'] for d in anti_exp_vals]
            if np.max(exp_val_list) <= 0:
                logger.info('All of the operators had <H_A> < 0, falling back')
                return current_ops + [new_op]
            else:
                max_ind = np.argmax(exp_val_list)
                new_op = anti_exp_vals[max_ind]['op']
                return current_ops + [new_op]
    
    def anti_commuting_part(self, operator: WeightedPauliOperator) -> WeightedPauliOperator:
        result = WeightedPauliOperator([])

        for c, p in self._hamiltonian.paulis:
            part = WeightedPauliOperator.from_list(paulis=[p], weights=[c])
            if part.anticommute_with(operator):
                result += part
        return result
    
    def compute_anti_exp_vals(self, circuit: QuantumCircuit):
        result_list = []
        for op in self._operator_pool.pool:
            part = self.anti_commuting_part(op)
            if part != WeightedPauliOperator([]):
                logger.info('Found candidate for anti-commutation: {}'.format(
                    op.print_details()
                ))
                result_list.append({
                    'part': part,
                    'op': op
                })
        exp_vals = multi_circuit_eval(
            circuit,
            op_list=[
                d['part']
                for d in result_list
            ],
            qi = self.quantum_instance,
            drop_dups = self._drop_duplicate_circuits
        )
        for i in range(len(result_list)):
            result_list[i]['mean'] = exp_vals[i][0].real
            result_list[i]['std'] = exp_vals[i][1]
            logger.info('Computed <H_A> for operator: {}'.format(
                result_list[i]['mean']
            ))
        return result_list

def get_Ha_Hc(op, **kwargs):
    term_list = kwargs['hp']
    term_e_list = kwargs['he']
    Ha_energy = 0
    Hc_energy = 0
    for i,term in enumerate(term_list):
        if not term.commute_with(op):
            Ha_energy = Ha_energy + term_e_list[i][0]
        else:
            Hc_energy = Hc_energy + term_e_list[i][0]
    return Ha_energy, Hc_energy




class Ha_max_OperatorSelector(OperatorSelector):

    def get_new_operator_list(self, term_list, term_e_list, current_ops, current_circuit):

        kwargs = {'hp': term_list, 'he': term_e_list}
        Ha_max_sq = -10000
        for i,op in enumerate(self._operator_pool.pool):
            if op.print_details()[:op.num_qubits] != current_ops[-1].print_details()[:op.num_qubits]:
                new_ha, new_hc = get_Ha_Hc(op, **kwargs)
                new_ha_sq = new_ha**2
                if new_ha_sq > Ha_max_sq:
                    Ha_max_sq = new_ha_sq
                    Ha_max_index = i
                    best_op = op
                    best_op_Ha_list = [new_ha]
                    best_op_Hc_list = [new_hc]
                    best_op_list = [op.print_details()[:op.num_qubits]]
                if new_ha_sq == Ha_max_sq:
                    Ha_max_sq = new_ha_sq
                    Ha_max_index = i
                    best_op = op
                    best_op_Ha_list.append(new_ha)
                    best_op_Hc_list.append(new_hc)
                    best_op_list.append(op.print_details()[:op.num_qubits])
        pool_copy = deepcopy(self._operator_pool)

        self._operator_pool = PauliPool.from_pauli_strings(best_op_list)

        grads, max_grad, evals = self._compute_gradients(current_circuit)
        self._coms = None
        min_energy = 10000
        for i,grad in enumerate(grads):
            energy = best_op_Hc_list[i] - np.sqrt(best_op_Ha_list[i]**2 + (grad[0]**2)/4)
            if energy < min_energy:
                min_energy = energy
                index = i

        new_op = self._operator_pool.pool[i]

        self._operator_pool = pool_copy

        return current_ops + [new_op]


class exclude_Ha_max_OperatorSelector(OperatorSelector):

    def get_new_operator_list(self, term_list, term_e_list, current_ops, current_circuit):

        kwargs = {'hp': term_list, 'he': term_e_list}
        Ha_max_sq = -10000
        for i,op in enumerate(self._operator_pool.pool):
            new_ha, new_hc = get_Ha_Hc(op, **kwargs)
            new_ha_sq = new_ha**2
            if new_ha_sq > Ha_max_sq:
                Ha_max_sq = new_ha_sq
                Ha_max_index = i
                best_op = op
                best_op_Ha_list = [new_ha]
                best_op_Hc_list = [new_hc]
                best_op_list = [op.print_details()[:op.num_qubits]]
            if new_ha_sq == Ha_max_sq:
                Ha_max_sq = new_ha_sq
                Ha_max_index = i
                best_op = op
                best_op_Ha_list.append(new_ha)
                best_op_Hc_list.append(new_hc)
                best_op_list.append(op.print_details()[:op.num_qubits])


        pool_copy = deepcopy(self._operator_pool)
        op_list = []
        for op in self._operator_pool.pool:
            if op.print_details()[:op.num_qubits] not in best_op_list:
                op_list.append(op.print_details()[:op.num_qubits])

        self._operator_pool = PauliPool.from_pauli_strings(op_list)

        grads, max_grad, evals = self._compute_gradients(current_circuit)
        self._coms = None
        max_grad= -100000
        print('good grads')
        for i,grad in enumerate(grads):
            print('grad', grad[0])
            print('max grad', max_grad)
            if abs(grad[0]) > max_grad:
                max_grad = abs(grad[0])
                index = i
        new_op = self._operator_pool.pool[i]

        self._operator_pool = PauliPool.from_pauli_strings(best_op_list)
        grads, max_grad, evals = self._compute_gradients(current_circuit)
        print('bad grads')
        for i, grad in enumerate(grads):
            print(grad[0])


        self._operator_pool = pool_copy



        return current_ops + [new_op]


class ADAPTQAOAOperatorSelector(ADAPTOperatorSelector):

    def get_new_operator_list(self, grad_list: List[Tuple[complex, complex]], current_ops, circuit: QuantumCircuit = None) -> Tuple[List[BaseOperator], float]:
        list_out = super().get_new_operator_list(grad_list, current_ops, circuit)
        list_out.append(self._hamiltonian)
        return list_out


class Approx_Ha_maxOperatorSelector(OperatorSelector):

    def get_new_operator_list(self, term_list, term_e_list, current_ops):
        num_qubits = self._hamiltonian.num_qubits
        weight_list = {'X': [0]*num_qubits, 'Y': [0]*num_qubits, 'Z': [0]*num_qubits, 'I': [0]*num_qubits}
        num_list = {'X': [0]*num_qubits, 'Y': [0]*num_qubits, 'Z': [0]*num_qubits, 'I': [0]*num_qubits}
        start_p = ''
        start_w = 0
        start_q = 0
        for i,term in enumerate(term_list):
            for j in range(0,num_qubits):
                single_p = term.print_details[j]
                weight_list[single_p][j] = weight_list[single_p][j] + term_e_list[i]
                num_list[single_p][j] = num_list[single_p][j] + 1 #this won't work, need to keep track of which ones.
                if weight_list[single_p][j] > start_w:
                    start_w = weight_list[single_p][j]
                    start_p = single_p
                    start_q = j

        initial_op = 'I'*num_qubits
        initila_op[start_q] = start_p
