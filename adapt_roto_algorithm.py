
import time

import copy
# allows user to create copies of objects as opposed
# to using = which redefines object,
# can be shallow or deep https://docs.python.org/2/library/copy.html \n
import logging
# allows for display of updates and errors while program
# is running, can log to window or
# file https://docs.python.org/3/library/logging.html \n
from typing import List, Tuple, Union
# allows import of List, Tuple data types and union operation
# this is part of std library no? \n

import networkx
# Package for study of structures and graphs https://networkx.github.io/ \n

import numpy as np
# computational modules for Python including matrices,
# arrays and operations on these. https://numpy.org/ \n

from qiskit import QuantumCircuit
# A allows for fast creation of QuantumRegister
# and ClassicalRegister class objects. \n

from qiskit.aqua import AquaError
# Is a class for errors raised by qiskit aqua.
# https://qiskit.org/documentation/apidoc/aqua/aqua.html \n

from qiskit.aqua import aqua_globals

from qiskit.aqua.algorithms import QuantumAlgorithm
# Basic class for quantum algorithms

from qiskit.aqua.algorithms import VQE
# The VQE algorithm class: VQE(operator, var_form, optimizer,
# initial_point=None, max_evals_grouped=1, aux_operators=None,
# callback=None, auto_conversion=True) \n

from qiskit.aqua.components.initial_states import InitialState, Zero, Custom
# InitialState, basic class for initializing quantum
# statevector, Zero state constructor, custom initial state constructor \n

from qiskit.aqua.components.optimizers import Optimizer
# Constructor for initializing optimization algorithm

from qiskit.aqua.components.variational_forms import VariationalForm
# Base class for variational forms

from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator
# operators class BaseOperator(basis=None, z2_symmetries=None, name=None),
# weighted pauli class, 2 term list, with weight and operator
# Baseoperator -
# https://qiskit.org/documentation/api/qiskit.aqua.operators.BaseOperator.html
# This is an abstract class, in fact many of these are, 
# more infor here: https://www.python-course.eu/python3_abstract_classes.php \\n

from qiskit.tools.parallel import parallel_map
# Allows for classical parallelization of mapping values to a function
# https://qiskit.org/documentation/_modules/qiskit/tools/parallel.html \n

from .adapt_variational_form import ADAPTVariationalForm, MixerLayer, CompositeVariationalForm
# From other file

from .operator_pool import OperatorPool, PauliPool
# From other file

from .adapt_algorithm import ADAPTVQE
#from other file

logger = logging.getLogger(__name__)

# Options:
#	1. ADAPT then ROTO: Find ansatz using normal ADAPT, then implement ROTOSOLVE at end for each operator/parameter pair already in ansatz 
#	2. ADAPT using ROTO: Use ROTOSOLVE on each operator/parameter pair as it's added, then iterate until final energy - allows for better op choice
#	3. ADAPT mix ROTO: Use ROTOSOLVE on each operator/parameter par as it's added, then classically optimize further
#		- if 2 is done with variable shots for optimal parameter measurement and variable optimizer max calls then can find best mix
#
#	Perhaps set up option in init to select which of these to implement?


#need new var form for each object as we need _current_operator_list + new op from pool, we'll put this guy through a parallelization
# current operator list saved as literal list with newest ops at end
def ROTO_variational_form(op, *args, **kwargs):
    if args:
        current_op_list = list(args)
    else:
        current_op_list = []
    if kwargs['initial state'] is not None:
        initial_state = kwargs['initial state']
    else:
        initial_state = None
    op_list = current_op_list + [op]
    bounds=[(-np.pi, +np.pi)]*len(op_list)
    vf = ADAPTVariationalForm(op_list, bounds, initial_state)
    return vf

# This is new ADAPTVQEROTO, currently is ADAPT using ROTO, full pseudocode to follow + ADAPT mx ROTO functionality.
class ADAPTVQEROTO(ADAPTVQE):
    def __init__(
            self,
            operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters= 10,
            grad_tol= 1e-3,
            energy_step_tol= 1e-5,
            max_evals_grouped=1,
            aux_operators=None,
            auto_conversion=True,
            use_zero_initial_parameters=False
        ): #most of these params no longer matter, should edit to remove necessity
        self._quantum_instance = None
        super().__init__(operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters = 10, #mximum number of iterations
            grad_tol = 1e-3, #stopping criteria if we were using gradient tolerance (we're not)
            max_evals_grouped=1,#unsure what this means
            aux_operators=None, #additional operators in pool
            auto_conversion=True, #when using VQE algorithm auto converts operators to best representaiton for backend
            use_zero_initial_parameters=False #Make your parameters start at zero before gradient based optimization
        ) #initialization from ADAPTVQE
        self.energy_step_tol = energy_step_tol #the stopping criteria for energy step
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.evaluations = 0

        if initial_state is None: #creates new initial state if none passed
            self.initial_state = Custom(hamiltonian.num_qubits, state='uniform')
        else:
            self.initial_state = initial_state

    def _test_energy_evaluation(self, var_form, *args):
        #we need to evaluate hamiltonian with var form as our wavefunction
        parameter_value_list = np.array(args)
        temporary_VQE = VQE(self.hamiltonian, var_form, self.vqe_optimizer, parameter_value_list, self.max_evals_grouped, self.aux_operators, None, self.auto_conversion)
        temporary_VQE._quantum_instance = self._quantum_instance
        energy = temporary_VQE._energy_evaluation(parameter_value_list)# make sure to only take 1st item of array for now as _energy_evaluatio returns array of mean energies, entry for each "parameter set"
        return energy

    def find_optim_param_energy(self) -> dict:
        #will need to see if faster sequential or parallel

        args = tuple(self._current_operator_list)
        kwargs = {'initial state': self.initial_state}
        var_form_list = list(parallel_map(
            ROTO_variational_form,
            self.operator_pool.pool,
            args,
            kwargs,
            num_processes=aqua_globals.num_processes #https://github.com/Qiskit/qiskit-aqua/pull/635
        ))  # type: List[variational_form] outputs list of variational form objects to be used as test ansatz's, one for each new possible operator
        curr_params = self.adapt_step_history['optimal_parameters']
        Optim_energy_array = []
        Optim_param_array = []
        if var_form_list:
            for i,form in enumerate(var_form_list): #use the ROTO optimization alg to find optim param, measure to find optimal energy.
                curr_params.append(0) #be careful with param def, use pi/4 instead of pi/2 bc benedetti define param = theta/2 and theta = pi/2
                energy_0 = self._test_energy_evaluation(form, *curr_params)
                del curr_params[-1]
                curr_params.append(np.pi/4)
                energy_pi4 = self._test_energy_evaluation(form, *curr_params)
                del curr_params[-1]
                curr_params.append(-np.pi/4)
                energy_negpi4 = self._test_energy_evaluation(form, *curr_params)
                del curr_params[-1]
                new_param = -np.pi/4 - np.arctan2((2*energy_0 - energy_pi4 - energy_negpi4),(energy_pi4 - energy_negpi4))
                if new_param >= np.pi:
                    new_param = new_param - 2*np.pi
                if new_param < -np.pi:
                    new_param = new_param + 2*np.pi
                Optim_param_array.append(new_param)
                curr_params.append(new_param)
                Optim_energy = self._test_energy_evaluation(form, *curr_params)
                del curr_params[-1]
                Optim_energy_array.append(Optim_energy)
                self.evaluations += 4
            Optim_param_pos = np.argmin(Optim_energy_array)
            min_energy = Optim_energy_array[Optim_param_pos]
            Optim_param = Optim_param_array[Optim_param_pos]
            Optim_operator = self.operator_pool.pool[Optim_param_pos]
            Optim_operator_name = self.operator_pool.pool[Optim_param_pos].print_details()
        else:
            min_energy = None
            Optim_param = None
            Optim_operator = None
            Optim_operator_name = None
        return {'Newly Minimized Energy': min_energy, 'Next Parameter value': Optim_param, 'Next Operator identity': Optim_operator, 'Next Operator Name': Optim_operator_name}
    def recent_energy_step(self):
        return abs(self.adapt_step_history['energy_history'][-1] - self.adapt_step_history['energy_history'][-2])
        #Our new run function, this will take time to edit
    def run(self, quantum_instance) -> dict: #we'll keep this structure of returning a dictionary, though maybe I'll return the full changelog,
        start = time.time()
        self._current_operator_list = []
        self._quantum_instance = quantum_instance
        iters = 0
        while iters <= 1:
            logger.info('Starting ADAPTROTO step {} of maximum {}'.format(iters, self.max_iters)) 
            New_minimizing_data = self.find_optim_param_energy()
            self._current_operator_list.append(New_minimizing_data['Next Operator identity'])
            self.adapt_step_history['optimal_parameters'].append(New_minimizing_data['Next Parameter value'])
            self.adapt_step_history['operators'].append(New_minimizing_data['Next Operator Name'])
            self.adapt_step_history['energy_history'].append(New_minimizing_data['Newly Minimized Energy'])
            logger.info('Finished ADAPTROTO step {} of maximum {} with energy {}'.format(iters, self.max_iters, self.adapt_step_history['energy_history'][-1]))
            iters += 1
        while iters <= self.max_iters and self.recent_energy_step() >= self.energy_step_tol:
            print(self.max_iters)
            logger.info('Starting ADAPTROTO step {} of maximum {}'.format(iters, self.max_iters)) 
            New_minimizing_data = self.find_optim_param_energy()
            self._current_operator_list.append(New_minimizing_data['Next Operator identity'])
            self.adapt_step_history['optimal_parameters'].append(New_minimizing_data['Next Parameter value'])
            self.adapt_step_history['operators'].append(New_minimizing_data['Next Operator Name'])
            self.adapt_step_history['energy_history'].append(New_minimizing_data['Newly Minimized Energy'])
            logger.info('Finished ADAPTROTO step {} of maximum {} with energy {}'.format(iters, self.max_iters, self.adapt_step_history['energy_history'][-1]))
            iters += 1
        logger.info('Final energy step is {} where tolerance is {}'.format(
            self.recent_energy_step(), self.energy_step_tol
        ))
        print('final iters: ', iters)
        eval_time = time.time() - start
        self.adapt_step_history.update({"Total Eval time": eval_time})
        return self.adapt_step_history #return final minimized energy list