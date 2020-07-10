import logging
from copy import deepcopy
from typing import List, Union, Dict
import pandas as pd

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
from operator_selector_new import OperatorSelector, multi_circuit_eval, Ha_max_OperatorSelector
from qiskit.aqua import aqua_globals, QuantumInstance


logger = logging.getLogger(__name__)

def convert_to_wpauli_list(term):
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
	ham_list = ham.paulis
	separated_ham_list = parallel_map(convert_to_wpauli_list, ham_list,  num_processes = aqua_globals.num_processes)

	return separated_ham_list


class ADAPT_maxH(IterativeVQE):
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

        self.ham_list = split_into_paulis(self.hamiltonian)

        if operator_selector is None: #need to change, should be roto
            self._operator_selector = Ha_max_OperatorSelector(
                self.hamiltonian,
                operator_pool=self.operator_pool,
                drop_duplicate_circuits=self._drop_duplicate_circuits,
            )
        else:
            self._operator_selector = operator_selector

        self._parameter_tolerance = parameter_tolerance

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
        new_op_info = self._operator_selector.get_new_operator_list(self.ham_list,last_result['expec list'], last_result['current_ops'])
        new_op_list = new_op_info


        if self.initial_parameters == 1:
            self.__new_param = new_op_info['roto param']

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
        expec_list, evals = multi_circuit_eval(result['current_circuit'], self.ham_list, qi = self._operator_selector.quantum_instance)
        result['num op choice evals'] = evals
        result['Hc_list'], result['Ha_list'] = self.operator_selector.sort_energies(expec_list)
        if self._compute_hessian:
            hessian = self._operator_selector._hessian(circuit=result['current_circuit'])
        else:
            hessian = None
        result['hessian'] = hessian

        return result

    @property
    def _new_param(self):
        output = [np.random.uniform(-np.pi, +np.pi)]
        return np.array(output)

    def variational_form(self, ops):
        return ADAPTVariationalForm(
            operator_pool=ops,
            bounds=[(-np.pi, +np.pi)] * len(ops),
            initial_state=self.initial_state,
            tolerance=self._parameter_tolerance
        )