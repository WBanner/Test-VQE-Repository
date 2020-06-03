"""
ADAPTROTO File
"""

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
from operator_selector_new import OperatorSelector, multi_circuit_eval
from qiskit.aqua import aqua_globals, QuantumInstance

logger = logging.getLogger(__name__)



def Generate_roto_op(op,*args, **kwargs):
    parameter = kwargs['parameter']
    ham = kwargs['ham']
    energy_step_tol = kwargs['energy_step_tol']
    Iden = WeightedPauliOperator.from_list(paulis=[Pauli.from_label('I' * ham.num_qubits)],
                                                weights=[1.0])
    psi = np.cos(parameter)*Iden + 1j*np.sin(parameter)*op
    psi_star = np.cos(parameter)*Iden - 1j*np.sin(parameter)*op
    return ((psi)*ham*(psi_star)).chop(threshold=energy_step_tol, copy=True)


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
            initial_parameters: Union[int, float] = 1,
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

        if operator_selector is None: #need to change, should be roto
            self._operator_selector = ROTOADAPTOperatorSelector(
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
        print('new kwargs')
        new_op_info = self._operator_selector.get_new_operator_list(last_result)
        del self._step_history[-1]['roto energy list']
        del self._step_history[-1]['roto parameter list']
        new_op_list = new_op_info['op list']


        if self.initial_parameters == 1:
            self.__new_param = new_op_info['roto param']

        var_form = self.variational_form(new_op_list)
        initial_point = np.concatenate((
            last_result['opt_params'],
            self._new_param
        ))
        print('finish new kwargs')
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
        print('start postprocess')
        result = super().post_process_result(result, vqe, last_result)
        result['current_ops'] = deepcopy(vqe._var_form._operator_pool)
        result['num op choice evals'], result['roto energy list'], result['roto parameter list'] = self._operator_selector.get_energy_param_lists(result)
        print('post process', self.step)
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


class ROTOADAPTOperatorSelector(OperatorSelector):
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
    def get_energy_param_lists(self, result):
        """
        	method: get_energy_param_lists
        	args:
        		result- the data for the recently calculated result
        	returns:
        		dict with number of energy evaluations required, array of optimized energies for each operator in pool,
        		 array of optimized parameters for the energy values in energy array
        """
        if self.parameters_per_step == 1:
            measured_energies = self._measure_energies(
            	result['current_circuit'],
            	result['current_ops']
            )
            optimal_array = self._get_optimal_array(result['energy'], measured_energies)
            return measured_energies['num evals'], optimal_array['energy array'], optimal_array['param array']
    def _measure_energies(self, wavefunc, *current_ops):
        """
        method: measure_energies
               finds the meausred energy for each new operator at pi/2 and -pi/2
        args: 
           circuit - the current optimal circuit
           current_ops - the current optimal operators
        returns:
           dictionary with: evaluation energy lists with parameter values at pi/2 and -pi/2
        """
        #measure energies (theta = 0, theta = pi/4, theta = -pi/4)
        args = []
        kwargs = {'ham': self._hamiltonian, 'energy_step_tol': self.energy_tol, 'parameter': np.pi/4}
        op_list_pi4 = list(parallel_map(
            Generate_roto_op,
            self._operator_pool.pool,
            args,
            kwargs
        ))
        print('created first op list')
        kwargs['parameter'] = -np.pi/4
        op_list_negpi4 = list(parallel_map(
            Generate_roto_op,
            self._operator_pool.pool,
            args,
            kwargs
        ))
        print('created second op list')
        op_list = op_list_pi4 + op_list_negpi4
        del op_list_pi4
        del op_list_negpi4
        E_list, evals = np.real(multi_circuit_eval(wavefunc, op_list, self.quantum_instance, self._drop_duplicate_circuits))
        del op_list
        E_list, E_list_std = list(zip(*E_list))
        print('created E list')
        cutoff = int(len(E_list)/2)
        Energy_pi4 = np.array(E_list[0:cutoff])
        Energy_negpi4 = np.array(E_list[cutoff:])
        return {'energy pi4': Energy_pi4, 'energy negpi4': Energy_negpi4, 'num evals': evals}
    def get_new_operator_list(self, last_result):
        """
        	method: get_new_operator_list
        		chooses new operator that minimizes the ansatz energy if added.
        	args:
        		last_result: a dict that stores the optimal energy list and optimal parameter list for all possible operators that could be added
        	returns:
        		dict with the new op list and a suggested new parameter based on the rotosolve optimization
        """
        #find minimum energy index
        Optim_param_pos = np.argmin(last_result['roto energy list'])
        min_energy = last_result['roto energy list'][Optim_param_pos]
        Optim_param = last_result['roto parameter list'][Optim_param_pos]
        #CPU has limit on smallest number to be calculated - looks like its somewhere around 1e-16
        #manually set this to zero as it should be zero.
        if min_energy > last_result['energy'] and abs(Optim_param) < 2e-16:
            Optim_param = 0
        #find optimum operator
        Optim_operator = self._operator_pool.pool[Optim_param_pos]
        return {'op list': last_result['current_ops'] + [Optim_operator], 'roto param': Optim_param}
    def _get_optimal_array(self, Energy_0, measured_energies):
        """
        	method: _get_optimal_array
        		determines the optimal (minimum) energies of any operator to be added and the new parameter value at which this occurs
        	args:
        		Energy_0 - the energy of the ansatz without a new operator added (new parameter = 0)
        		measured_energies - dict with 2 lists of measurement results at new parameter = pi/2 (or pi/4) and new parameter = -pi/2 (-pi/4)
        	returns:
        		dict with param array: array of the parameters at their optimal positions for a given new operator
        				  energy array: array of minimized energies for each operator in pool (occurs at the corresponding parameter in parameter array)
        """
        Energy_pi4 = np.array(measured_energies['energy pi4'])
        Energy_negpi4 = np.array(measured_energies['energy negpi4'])
        A = np.empty(0)
        #calculate minimum energy + A,B, and C from measured energies
        B = np.arctan2((-Energy_negpi4 - Energy_pi4 + 2*Energy_0).astype(float), (Energy_pi4 - Energy_negpi4).astype(float))
        Optim_param_array = (-B - np.pi/2)/2
        X = np.sin(B)
        Y = np.sin(B + np.pi/2)
        C = 0.5*(Energy_pi4 + Energy_negpi4)
        A = (Energy_0 - C)/X #for some reason below didn't work -> ruins previous results? why then did previous results work?
        """
        for i in range(0,len(Energy_negpi4)):
            if Y[i] != 0:
                A = np.append(A, (Energy_pi4[i] - C[-1])/Y[i])
            else:
                A = np.append(A, (Energy_0 - C[-1])/X[i])
        """
        Optim_energy_array = C - A
        #print(Optim_energy_array)
        return {'param array': Optim_param_array, 'energy array': Optim_energy_array}



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
def split_into_paulis(ham):
    ham_details = ham.print_details()
    ham_list = ham_details.split('\n')
    pauli_list = [0]*(len(ham_list)-1) #exclude last entry of ham list bc is just blank
    name_list = [0]*(len(ham_list)-1)
    weight_list = [0]*(len(ham_list)-1)
    for counter in range(0, len(ham_list)-1,1):
        pauli = Pauli.from_label(ham_list[counter][:4])
        name_list[counter] = WeightedPauliOperator.from_list([pauli]).print_details()
        weight_list[counter] = complex(ham_list[counter][6:-1])
        pauli_list[counter] = WeightedPauliOperator.from_list([pauli])
    return weight_list, pauli_list, name_list



class efficientROTOADAPTOperatorSelector(ROTOADAPTOperatorSelector):
    def __init__(
            self, 
            hamiltonian, 
            operator_pool: OperatorPool, 
            drop_duplicate_circuits: bool = True, 
            energy_tol: float = None,
            parameters_per_step: int = 1):

        super().__init__(hamiltonian, 
        operator_pool, 
        drop_duplicate_circuits, 
        energy_tol,
        parameters_per_step)
    def _reconstruct_single_energy_expression(self, op, ham = None, op_finding = False):
        expecs = pd.read_csv('Pauli_values.csv')
        expecs = expecs.to_dict()
        if op_finding == True:
            ham_pauli_list = self.ham_pauli_list
            ham_weight_list = self.ham_weight_list
            
        else:
            ham_weight_list, ham_pauli_list, ham_name_list = split_into_paulis(ham)
        kwargs = {'op_1': op}
        args = []
        H_c_eng = 0
        H_c_name_list = []
        H_a_name_list = []
        H_c_weight_list = []
        H_a_eng = 0
        H_a_op = 0*self._operator_pool.pool[0]
        for i in range(0,(len(ham_pauli_list))):
            for j in range(0, (len(expecs['exp vals']))):
                if ham_name_list[i] == expecs['names'][j]:
                    if does_commute(ham_pauli_list[i], kwargs):
                        H_c_eng = H_c_eng + ham_weight_list[i]*complex(expecs['exp vals'][j]) #will need to be changed to make sure are in same order
                        H_c_name_list.append(ham_name_list[i])
                        H_c_weight_list.append(ham_weight_list[i]*complex(expecs['exp vals'][j]))
                    else:
                        H_a_eng = H_a_eng + ham_weight_list[i]*complex(expecs['exp vals'][j])
                        H_a_name_list.append(ham_name_list[i])
                        H_a_op = H_a_op + ham_pauli_list[i]*ham_weight_list[i]
        comm = find_commutator(H_a_op, kwargs)
        comm_weight_list, comm_pauli_list, comm_name_list = split_into_paulis(comm)
        comm_eng = 0
        for i in range(0, (len(comm_name_list))):
            for j in range(0, (len(expecs['exp vals']))):
                if comm_name_list[i] == expecs['names'][j]: 
                    comm_eng = complex(expecs['exp vals'][j])*comm_weight_list[i] + comm_eng
        return H_c_eng, H_a_eng, comm_eng
    def _get_optimal_array(self, Energy_0, measured_energies):
        Energy_pi4 = np.array(measured_energies['energy pi4'])
        Energy_negpi4 = np.array(measured_energies['energy negpi4'])
        old_A = np.empty(0)
        #calculate minimum energy + A,B, and C from measured energies
        old_B = np.arctan2((-Energy_negpi4 - Energy_pi4 + 2*Energy_0).astype(float), (Energy_pi4 - Energy_negpi4).astype(float))
        X = np.sin(old_B)
        Y = np.sin(old_B + np.pi/2)
        old_C = 0.5*(Energy_pi4 + Energy_negpi4)
        old_A = (Energy_0 - old_C)/X
        old_H_a_eng = 0.5*(old_A*X - old_A*np.sin(np.pi + old_B))
        old_optim_energy = old_C - old_A
        A = []
        B = []
        C = []
        for op in self._operator_pool.pool:
            A_single,B_single,C_single = self._reconstruct_single_energy_expression(op, self._hamiltonian, op_finding = False)
            A.append(A_single)
            B.append(B_single)
            C.append(C_single)
        A = np.array(A)
        B = np.array(B)
        C = np.array(C)
        Amp = np.sqrt(np.square(B)+np.square(C)/4)
        Optim_energy_array = A - Amp
        print(Optim_energy_array)
        bottom_arr = np.real(1j*C)
        top_arr = np.real(2*B)
        Optim_param_array = -np.arctan2(top_arr, bottom_arr) - np.pi/2
        return {'param array': Optim_param_array, 'energy array': Optim_energy_array}


