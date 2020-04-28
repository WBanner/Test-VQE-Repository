
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

def evaluate_energy(var_form, *args, **kwargs):
    parameter_value_list = np.array(args)
    temporary_VQE = VQE(kwargs['ham'], var_form, kwargs['optimizer'], parameter_value_list, kwargs['max_evals_grouped'], kwargs['aux_operators'], None, kwargs['auto_conversion'])
    temporary_VQE._quantum_instance = kwargs['quantum_instance']
    energy = temporary_VQE._energy_evaluation(parameter_value_list)
    return energy

# This is new ADAPTVQEROTO, currently is ADAPT using ROTO, full pseudocode to follow + ADAPT mx ROTO functionality.
class ADAPTVQEROTO(ADAPTVQE):
    def __init__(
            self,
            operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters,
            energy_step_tol= 1e-8,
            auto_conversion=True,
            use_zero_initial_parameters=False,
            postprocessing = False,
            include_parameter = True
        ): #most of these params no longer matter, should edit to remove necessity
        self._quantum_instance = None
        super().__init__(operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters, #mximum number of iterations
            grad_tol = 1e-3, #stopping criteria if we were using gradient tolerance (we're not)
            max_evals_grouped=1,#unsure what this means
            aux_operators=None, #additional operators in pool
            auto_conversion=True, #when using VQE algorithm auto converts operators to best representaiton for backend
            use_zero_initial_parameters=False #Make your parameters start at zero before gradient based optimization
        ) #initialization from ADAPTVQE
        self.energy_step_tol = energy_step_tol #the stopping criteria for energy step
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.postprocessing = postprocessing
        self.include_parameter = include_parameter

        if initial_state is None: #creates new initial state if none passed
            self.initial_state = Zero(num_qubits=operator_pool.num_qubits) #Custom(hamiltonian.num_qubits, state='uniform')
        else:
            self.initial_state = initial_state

        #clean up old dictionary
        del self.adapt_step_history['gradient_list']
        del self.adapt_step_history['max_gradient']
        #del self.adapt_step_history['vqe_ret']

        #self.adapt_step_history.update({'Total Eval Time': 0})
        #self.adapt_step_history.update({'Total num Evals': 0})
        self.adapt_step_history.update({'Total number energy iterations': 0}) 
        self.adapt_step_history.update({'Max Energy Step': 0})


    def find_optim_param_energy(self, preferred_op = None, preferred_op_mode = False) -> dict:
        #will need to see if faster sequential or parallel
       	args = tuple(self._current_operator_list)
        kwargs = {'initial state': self.initial_state}
        if preferred_op_mode:
            var_form_list = []
            var_form_list.append(ROTO_variational_form(preferred_op, *args, **kwargs))
        else:
            var_form_list = list(parallel_map(
                ROTO_variational_form,
                self.operator_pool.pool,
                args,
                kwargs,
                num_processes=aqua_globals.num_processes #https://github.com/Qiskit/qiskit-aqua/pull/635
            ))  # type: List[variational_form] outputs list of variational form objects to be used as test ansatz's, one for each new possible operator
        curr_params = []
        if self.adapt_step_history['Total number energy iterations'] > 0:
        	curr_params = list(self.adapt_step_history['optimal_parameters'][-1])
        A = np.empty(0)
        C = np.empty(0)
        curr_params.append(0) #be careful with param def, use pi/4 instead of pi/2 bc benedetti define param = theta/2 and theta = pi/2
        kwargs = {'ham': self.hamiltonian, 'optimizer': self.vqe_optimizer, 'max_evals_grouped': self.max_evals_grouped, 'aux_operators': self.aux_operators, 'auto_conversion': self.auto_conversion, 'quantum_instance': self.quantum_instance}
        Energy_0 = evaluate_energy(var_form_list[0], *curr_params, **kwargs) #only need to do this once.
        del curr_params[-1]
        curr_params.append(np.pi/4)
        args = tuple(curr_params)
        Energy_pi4 = np.array(parallel_map(
            evaluate_energy,
            var_form_list,
            args,
            kwargs,
            num_processes=aqua_globals.num_processes #https://github.com/Qiskit/qiskit-aqua/pull/635
        ))
        del curr_params[-1]
        curr_params.append(-np.pi/4)
        args = tuple(curr_params)
        Energy_negpi4 = np.array(parallel_map(
            evaluate_energy,
            var_form_list,
            args,
            kwargs,
            num_processes=aqua_globals.num_processes #https://github.com/Qiskit/qiskit-aqua/pull/635
        )) 
        del curr_params[-1]
        B = np.arctan2(( -Energy_negpi4 - Energy_pi4 + 2*Energy_0),(Energy_pi4 - Energy_negpi4))
        Optim_param_array = (-B - np.pi/2)/2
        X = np.sin(B)
        Y = np.sin(B + np.pi/2)
        Z = np.sin(B - np.pi/2)
        for i in range(0,(len(var_form_list) -1)):
            if Y[i] != 0:
                C = np.append(C, (Energy_0-Energy_pi4[i]*(X[i]/Y[i]))/(1-X[i]/Y[i]))
                A = np.append(A, (Energy_pi4[i] - C[-1])/Y[i])
            else:
                C = np.append(C, Energy_pi4[i])
                A = np.append(A, (Energy_0 - C[-1])/X[i])
        Optim_energy_array = C - A
        Optim_param_pos = np.argmin(Optim_energy_array)
        min_energy = Optim_energy_array[Optim_param_pos]
        Optim_param = Optim_param_array[Optim_param_pos]
        if min_energy > Energy_0 and abs(Optim_param) < 2e-16:
            Optim_param = 0
            min_energy = Energy_0
        Optim_operator = self.operator_pool.pool[Optim_param_pos]
        Optim_operator_name = self.operator_pool.pool[Optim_param_pos].print_details()
        self.adapt_step_history['Total num evals'] += 2*len(var_form_list) + 1

        return {'Newly Minimized Energy': min_energy, 'Next Parameter value': Optim_param, 
         'Next Operator identity': Optim_operator, 'Next Operator Name': Optim_operator_name, 'A': A[Optim_param_pos], 'B': B[Optim_param_pos], 'C': C[Optim_param_pos]}
    def recent_energy_step(self):
        return abs(self.adapt_step_history['energy_history'][-1] - self.adapt_step_history['energy_history'][-2])
        #Our new run function, this will take time to edit
    def run(self, quantum_instance) -> dict: #we'll keep this structure of returning a dictionary, though maybe I'll return the full changelog,
        start = time.time()
        self._current_operator_list = []
        self._quantum_instance = quantum_instance
        while self.adapt_step_history['Total number energy iterations'] <= 1:
            logger.info('Starting ADAPTROTO step {} of maximum {}'.format(self.adapt_step_history['Total number energy iterations'], self.max_iters)) 
            New_minimizing_data = self.find_optim_param_energy()
            if self.adapt_step_history['Total number energy iterations'] > 1: #and New_minimizing_data['Newly Minimized Energy'] > self.adapt_step_history['energy_history'][-1]:
                break
            self._current_operator_list.append(New_minimizing_data['Next Operator identity'])
            if self.include_parameter == True:
                new_parameter = float(New_minimizing_data['Next Parameter value'])
            else:
                new_parameter = 0
            if self.adapt_step_history['Total number energy iterations'] == 0:
                self.adapt_step_history['optimal_parameters'].append([new_parameter])
            else:
                new_list = self.adapt_step_history['optimal_parameters'][-1] + [new_parameter]
                self.adapt_step_history['optimal_parameters'].append(new_list)
            self.adapt_step_history['operators'].append(New_minimizing_data['Next Operator Name'])
            if self.postprocessing and self.adapt_step_history['Total number energy iterations'] > 0:
                vqe_rotosolve_result = self._vqe_run(self._current_operator_list,np.array(self.adapt_step_history['optimal_parameters'][-1]))
                self.adapt_step_history['optimal_parameters'].append(list(vqe_rotosolve_result['optimal_params']))
                self.adapt_step_history['energy_history'].append(vqe_rotosolve_result['_ret']['energy'])
                self.adapt_step_history['Total num evals'] += vqe_rotosolve_result['_ret']['num_optimizer_evals']
            else:
                self.adapt_step_history['energy_history'].append(New_minimizing_data['Newly Minimized Energy'])
            logger.info('Finished ADAPTROTO step {} of maximum {} with energy {}'.format(self.adapt_step_history['Total number energy iterations'], self.max_iters, self.adapt_step_history['energy_history'][-1]))
            self.adapt_step_history['Total number energy iterations'] += 1
            print(self.adapt_step_history['Total number energy iterations'])

        if self.recent_energy_step() > self.adapt_step_history['Max Energy Step']:
            self.adapt_step_history['Max Energy Step'] = self.recent_energy_step()

        while self.adapt_step_history['Total number energy iterations'] <= self.max_iters: #and self.recent_energy_step() >= self.energy_step_tol:
            logger.info('Starting ADAPTROTO step {} of maximum {}'.format(self.adapt_step_history['Total number energy iterations'], self.max_iters)) 
            New_minimizing_data = self.find_optim_param_energy()
            #if New_minimizing_data['Newly Minimized Energy'] > self.adapt_step_history['energy_history'][-1]:
             #   break
            self._current_operator_list.append(New_minimizing_data['Next Operator identity'])
            if self.include_parameter == True:
                new_parameter = float(New_minimizing_data['Next Parameter value'])
            else:
                new_parameter = 0
            self.adapt_step_history['optimal_parameters'].append(self.adapt_step_history['optimal_parameters'][-1] + [new_parameter])
            self.adapt_step_history['operators'].append(New_minimizing_data['Next Operator Name'])
            if self.postprocessing:
                vqe_rotosolve_result = self._vqe_run(self._current_operator_list, np.array(self.adapt_step_history['optimal_parameters'][-1]))
                self.adapt_step_history['optimal_parameters'].append(list(vqe_rotosolve_result['optimal_params']))
                self.adapt_step_history['energy_history'].append(vqe_rotosolve_result['_ret']['energy'])
                self.adapt_step_history['Total num evals'] += vqe_rotosolve_result['_ret']['num_optimizer_evals']
            else:
                self.adapt_step_history['energy_history'].append(New_minimizing_data['Newly Minimized Energy'])
            logger.info('Finished ADAPTROTO step {} of maximum {} with energy {}'.format(self.adapt_step_history['Total number energy iterations'], self.max_iters, self.adapt_step_history['energy_history'][-1]))
            self.adapt_step_history['Total number energy iterations'] += 1
            print(self.adapt_step_history['Total number energy iterations'])
            if self.recent_energy_step() > self.adapt_step_history['Max Energy Step']:
                self.adapt_step_history['Max Energy Step'] = self.recent_energy_step()

        logger.info('Final energy step is {} where tolerance is {}'.format(
            self.recent_energy_step(), self.energy_step_tol
        ))
        eval_time = time.time() - start
        self.adapt_step_history['Total Eval Time'] = eval_time
        return self.adapt_step_history #return final minimized energy list


        #This is a test to do 2 parameter optimization/parameter space mapping and min value estimation.
class ROTOEXTENDED(ADAPTVQEROTO):
        def __init__(
            self,
            operator_pool,
            initial_state,
            vqe_optimizer,
            hamiltonian,
            max_iters,
            auto_conversion=True,
            use_zero_initial_parameters=False
        ):
	        super().__init__(
	            operator_pool,
	            initial_state,
	            vqe_optimizer,
	            hamiltonian,
	            max_iters,
	            energy_step_tol= 1e-5,
	            auto_conversion=True,
	            use_zero_initial_parameters=False
	        )

        def run(self, quantum_instance):
            start = time.time()
            self._current_operator_list = []
            self._quantum_instance = quantum_instance
            A_list = []
            B_list = []
            C_list = []
            loc_list = [-np.pi/4, 0, np.pi/4]
            while self.adapt_step_history['Total number energy iterations'] <= 1:
                New_minimizing_data = self.find_optim_param_energy()
                self._current_operator_list.append(New_minimizing_data['Next Operator identity'])
                self.adapt_step_history['optimal_parameters'].append(New_minimizing_data['Next Parameter value'])
                self.adapt_step_history['operators'].append(New_minimizing_data['Next Operator Name'])
                self.adapt_step_history['energy_history'].append(New_minimizing_data['Newly Minimized Energy'])
                self.adapt_step_history['Total number energy iterations'] += 1

            final_op = copy.deepcopy(self._current_operator_list[-1])
            del self._current_operator_list[-1]
            opt_param_1 = self.adapt_step_history['optimal_parameters'][-2]
            opt_param_2 = self.adapt_step_history['optimal_parameters'][-1]
            del self.adapt_step_history['optimal_parameters'][-1]

            for location in loc_list:
	            self.adapt_step_history['optimal_parameters'][-1] = location
	            loc_data = self.find_optim_param_energy(final_op, True)
	            A_list.append(loc_data['A'])
	            B_list.append(loc_data['B'])
	            C_list.append(loc_data['C'])

            self._current_operator_list.append(final_op)
            self.adapt_step_history['optimal_parameters'].append(opt_param_2)
            self.adapt_step_history['optimal_parameters'][-2] = opt_param_1

            #now need to reconstruct 2D parameter space from these equations
            #I can do it but it's gross, can get eqs for X,Y,Z in terms of p2 for E = Xsin(p1 + Y) + Z
            #need to find where Z-X is at a minimum. X always positive fyi and it's easy to tell where Z is pos and neg (if occurs)
            #only need 4 additional measurements to get Z fyi, thouh +1 to get the optimum 1st param
            #we could look at new choice criteria - where Z minimized, not sure how complicated for 3 or more params
            #could look at Y variation, if Y varies little then ADAPT should be good.
            #Unfortunately Y takes all 6 additional measurements.

            eval_time = time.time() - start
            self.adapt_step_history['Total Eval Time'] = eval_time
            self.adapt_step_history.update({'A': A_list})
            self.adapt_step_history.update({'B': B_list})
            self.adapt_step_history.update({'C': C_list})
            return self.adapt_step_history
