# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

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

from qiskit.aqua.operators import BaseOperator, WeightedPauliOperator, MatrixOperator
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

from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator

logger = logging.getLogger(__name__)


#from George
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

#from george
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

def Generate_op(op,*args, **kwargs):
    parameter = kwargs['parameter']
    ham = kwargs['ham']
    energy_step_tol = kwargs['energy_step_tol']
    mat = np.identity(2**op.num_qubits)
    Iden = to_weighted_pauli_operator(MatrixOperator(mat)) #creates random hamiltonian from random matrix "mat"
    #return np.exp(1j*parameter*op)*ham*np.exp(-1j*parameter*op).chop(threshold=energy_step_tol, copy=True)
    return (np.cos(parameter)*Iden + 1j*np.sin(parameter)*op)*ham*(np.cos(parameter)*Iden - 1j*np.sin(parameter)*op).chop(threshold=energy_step_tol, copy=True)

def _circ_eval(op, **kwargs):
    return op.construct_evaluation_circuit(**kwargs)

def _compute_energy(op, **kwargs):
    return op.evaluate_with_result(**kwargs)

def _hash(circ):
    return hash(str(circ))

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


    def find_optim_param_energy(self, preferred_op = None, preferred_op_mode = False,  previous_circuit = None) -> dict:
        """method: find_optim_param_energy
               finds the optimum operator+parameter pair for the next iteration of the ansatz
        args: 
           preferred_op - an operator (weightedpaulioperator) predetermined to be the next operator in the ansatz 
                           (essentially converts this method to rotosolve instead of rotoselect)
           preferred_op_mode - a flag (boolean) to choose whether or not preferred op should be used
        
        returns:
           dictionary with: optimal operator, name of optimal operator (for output purposes), optimal parameter,
                            optimal energy, A, B, and C for Asin(theta + B) + C
        """
        A = np.empty(0)
        C = np.empty(0)
        final_circs = []
        E_list = []
        #measure energies (theta = 0, theta = pi/4, theta = -pi/4)

        if(self.adapt_step_history['Total number energy iterations'] == 0):
            wavefunc = self.initial_state.construct_circuit()
            Energy_0_circ = self.hamiltonian.construct_evaluation_circuit(wavefunc, True)
            result = self.quantum_instance.execute(Energy_0_circ)
            Energy_0 = np.real(self.hamiltonian.evaluate_with_result(result, True)[0])
            self.adapt_step_history['Total num evals'] += 1
        else:
            op_list = self._current_operator_list
            bounds=[(-np.pi, +np.pi)]*len(op_list)
            vf = ADAPTVariationalForm(op_list, bounds, self.initial_state)
            wavefunc = vf.construct_circuit(self.adapt_step_history['optimal_parameters'][-1])
            Energy_0 = self.adapt_step_history['energy_history'][-1]

        if preferred_op_mode:
            pool = preferred_op
        else:
            pool = self.operator_pool.pool

        args = []
        kwargs = {'ham': self.hamiltonian, 'energy_step_tol': self.energy_step_tol, 'parameter': np.pi/4}
        op_list_pi4 = list(parallel_map(
            Generate_op,
            pool,
            args,
            kwargs
            ))
        kwargs['parameter'] = -np.pi/4
        op_list_negpi4 = list(parallel_map(
            Generate_op,
            self.operator_pool.pool,
            args,
            kwargs
            ))

        op_list = op_list_pi4 + op_list_negpi4

        kwargs = {'statevector_mode': self.quantum_instance.is_statevector}
        total_evaluation_circuits = list(parallel_map(
            _circ_eval,
            op_list,
            task_kwargs={**kwargs, 'wave_function': wavefunc},
            num_processes=aqua_globals.num_processes
        ))

        total_evaluation_circuits = [item for sublist in total_evaluation_circuits for item in sublist]
        logger.info('Removing duplicate circuits')

        for circ in total_evaluation_circuits:
            if not fast_circuit_inclusion(circ, final_circs):
                final_circs.append(circ)
        result = self.quantum_instance.execute(final_circs)

        Energies = list(parallel_map(
            _compute_energy,
            op_list,
            task_kwargs={**kwargs, 'result': result},
            num_processes=aqua_globals.num_processes
            ))

        for entry in Energies:
            E_list.append(np.real(entry[0]))

        cutoff = int(len(E_list)/2)
        Energy_pi4 = np.array(E_list[0:cutoff])
        Energy_negpi4 = np.array(E_list[cutoff:])
        #calculate minimum energy + A,B, and C from measured energies
        B = np.arctan2(( -Energy_negpi4 - Energy_pi4 + 2*Energy_0),(Energy_pi4 - Energy_negpi4))
        Optim_param_array = (-B - np.pi/2)/2
        X = np.sin(B)
        Y = np.sin(B + np.pi/2)
        Z = np.sin(B - np.pi/2)
        for i in range(0,(len(Energy_negpi4)-1)):
            if Y[i] != 0:
                C = np.append(C, (Energy_0-Energy_pi4[i]*(X[i]/Y[i]))/(1-X[i]/Y[i]))
                A = np.append(A, (Energy_pi4[i] - C[-1])/Y[i])
            else:
                C = np.append(C, Energy_pi4[i])
                A = np.append(A, (Energy_0 - C[-1])/X[i])
        Optim_energy_array = C - A
        #find minimum energy index
        Optim_param_pos = np.argmin(Optim_energy_array)
        min_energy = Optim_energy_array[Optim_param_pos]
        Optim_param = Optim_param_array[Optim_param_pos]

        #CPU has limit on smallest number to be calculated - looks like its somewhere around 1e-16
        #manually set this to zero as it should be zero.
        if min_energy > Energy_0 and abs(Optim_param) < 2e-16:
            Optim_param = 0
            min_energy = Energy_0

        #find optimum operator
        Optim_operator = self.operator_pool.pool[Optim_param_pos]
        Optim_operator_name = self.operator_pool.pool[Optim_param_pos].print_details()

        #keep track of number of quantum evaluations
        self.adapt_step_history['Total num evals'] += len(final_circs)

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
