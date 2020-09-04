"""
ADAPT variational algorithm.

Grimsley, H.R., Economou, S.E., Barnes, E. et al. An adaptive variational
algorithm for exact molecular simulations on a quantum computer. Nat Commun 10,
3007 (2019). https://doi.org/10.1038/s41467-019-10988-2

qubit-ADAPT-VQE: An adaptive algorithm for constructing hardware-efficient
ansatze on a quantum processor
Ho Lun Tang, Edwin Barnes, Harper R. Grimsley, Nicholas J. Mayhall, Sophia E. Economou
https://arxiv.org/abs/1911.10205
"""
import logging
from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.components.initial_states import InitialState, Zero, Custom
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
from qiskit.quantum_info import Pauli

from qisresearch.adapt.adapt_variational_form import ADAPTVariationalForm
from qisresearch.adapt.operator_pool import OperatorPool
from qisresearch.vqa import DummyOptimizer
from qisresearch.i_vqe.callbacks import MaxGradientStopper
from iterative_new import IterativeVQE
from operator_selector_new import OperatorSelector, ADAPTOperatorSelector, ADAPTQAOAOperatorSelector, AntiCommutingSelector

logger = logging.getLogger(__name__)


class ADAPTVQE(IterativeVQE):
    """Create an instance of the ADAPT-VQE algorithm.

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
    grad_tol : float
        If the maximum gradient at any step is below this threshold, then the
        algorithm will terminate.
    max_evals_grouped : int
        See documentation for `qiskit.aqua.algorithm.VQE`
    aux_operators : List[Operator]
        See documentation for `qiskit.aqua.algorithm.VQE`
    auto_conversion : bool
        See documentation for `qiskit.aqua.algorithm.VQE`
    use_zero_initial_parameters : Union[bool, float]
        If `True`, then the initial parameter for each new layer is `0`. If it
        is a `float`, then instead use that `float`.
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
        Whether or not to compute the Hessian at each layer of ADAPT. The Hessian
        is defined by the expectation value of the double commutator `[[H, P], Q]`
        for operators `P` and `Q`.

    Attributes
    ----------
    commutators : List[Operator]
        The commutators of the Hamiltonian with each of the elements in the pool.
        Used for gradient evaluation.
    """

    CONFIGURATION = {
        'name': 'RotoADAPTVQE',
        'description': 'RotoADAPT-VQE Algorithm',
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
            use_zero_initial_parameters: Union[bool, float] = True,
            callback=None,
            step_callbacks=[],
            drop_duplicate_circuits=True,
            return_best_result: bool = False,
            parameter_tolerance=None,
            compute_hessian: bool = False,
            operator_selector: OperatorSelector = None
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
        self.grad_tol = grad_tol
        self.max_evals_grouped = max_evals_grouped
        self.aux_operators = aux_operators
        self.auto_conversion = auto_conversion
        self._compute_hessian = compute_hessian
        self._drop_duplicate_circuits = drop_duplicate_circuits
        self.callback = callback
        self.step_callbacks = step_callbacks

        if operator_selector is None:
            self._operator_selector = ADAPTOperatorSelector(
                self.hamiltonian,
                operator_pool=self.operator_pool,
                drop_duplicate_circuits=self._drop_duplicate_circuits
            )
        else:
            self._operator_selector = operator_selector

        if type(use_zero_initial_parameters) is bool:
            self.use_zero_initial_parameters = use_zero_initial_parameters
            self.__new_par = 0.0
        elif type(use_zero_initial_parameters) is float:
            self.use_zero_initial_parameters = True
            self.__new_par = use_zero_initial_parameters
        else:
            raise ValueError('Invalid option for new parameters supplied: {}'.format(use_zero_initial_parameters))

        self.parameters_per_step = 1
        self._coms = None
        self._dcoms = None
        self._parameter_tolerance = parameter_tolerance

        if len(self.step_callbacks) == 0:
            self.step_callbacks.append(MaxGradientStopper(self.grad_tol))

    def _is_converged(self) -> bool:
        if self.step > self.max_iters:
            logger.info('Algorithm converged because max iterations ({}) reached'.format(self.max_iters))
            return True
        else:
            return False

    def first_vqe_kwargs(self) -> Dict:
        # This works for now, but always produces one extra parameter.
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
        new_op_list = self._operator_selector.get_new_operator_list(
            last_result['grad_list'],
            last_result['current_ops'],
            circuit=last_result['current_circuit']
        )
        print(new_op_list[-1].print_details())

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
            'auto_conversion': self.auto_conversion
        }

    def post_process_result(self, result, vqe, last_result) -> Dict:
        result = super().post_process_result(result, vqe, last_result)
        grad_list, max_grad, evals = self._operator_selector._compute_gradients(circuit = result['current_circuit']) #will added evals
        result['current_ops'] = vqe._var_form._operator_pool
        result['max_grad'] = max_grad
        result['grad_list'] = grad_list
        result['num op choice evals'] = evals #will added evals
        print(result['energy'])

        if self._compute_hessian:
            hessian = self._operator_selector._hessian(circuit=result['current_circuit'])
        else:
            hessian = None
        result['hessian'] = hessian

        return result

    @property
    def _new_param(self):
        if self.use_zero_initial_parameters:
            output = [self.__new_par for i in range(self.parameters_per_step)]
        else:
            output = [np.random.uniform(-np.pi, +np.pi) for i in range(self.parameters_per_step)]
        return np.array(output)

    def variational_form(self, ops):
        return ADAPTVariationalForm(
            operator_pool=ops,
            bounds=[(-np.pi, +np.pi)] * len(ops),
            initial_state=self.initial_state,
            tolerance=self._parameter_tolerance
        )


class ADAPTQAOA(ADAPTVQE):
    """This algorithm uses the structure of the QAOA ansatz (alternating a cost
    Hamiltonian with a mixer) with the ADAPT strategy. The mixer layer is determined
    at each layer by calculating the gradients with respect to elements in the pool
    as in the `qisresearch.i_vqe.adapt.ADAPTVQE` algorithm.
    """

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
            grad_tol: float = 1e-3,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            use_zero_initial_parameters: Union[bool, float] = True,
            callback=None,
            step_callbacks=[],
            drop_duplicate_circuits=True,
            return_best_result: bool = False,
            parameter_tolerance=None,
            compute_hessian: bool = False
    ):
        """Creating instances of this class is identical to that of
        `qisresearch.i_vqe.adapt.ADAPTVQE`.
        """
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
            use_zero_initial_parameters,
            callback,
            step_callbacks,
            drop_duplicate_circuits,
            return_best_result,
            parameter_tolerance,
            compute_hessian,
            operator_selector=ADAPTQAOAOperatorSelector(
                hamiltonian,
                operator_pool,
                drop_duplicate_circuits
            )
        )

        self.parameters_per_step = 2
        self.first_step = True

        if initial_state is None:
            self.initial_state = Custom(hamiltonian.num_qubits, state='uniform')
        else:
            self.initial_state = initial_state


