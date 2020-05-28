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

import logging
import functools

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator, WeightedPauliOperator,
                                   MatrixOperator, op_converter)
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_provider)

logger = logging.getLogger(__name__)


[docs]class VQE(VQAlgorithm):
    """
    The Variational Quantum Eigensolver algorithm.

    See https://arxiv.org/abs/1304.3061
    """

    CONFIGURATION = {
        'name': 'VQE',
        'description': 'VQE Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'vqe_schema',
            'type': 'object',
            'properties': {
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'L_BFGS_B'
                },
            },
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'RYRZ'
                },
            },
        ],
    }
##################################################################################################################################################################
    def __init__(self, operator, var_form, optimizer,
                 initial_point=None, max_evals_grouped=1, aux_operators=None, callback=None,
                 auto_conversion=True): 
        """Constructor.

        Args:
            operator (BaseOperator): Qubit operator
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
            initial_point (numpy.ndarray): optimizer initial point.
            max_evals_grouped (int): max number of evaluations performed simultaneously
            aux_operators (list[BaseOperator]): Auxiliary operators to be evaluated
                                                at each eigenvalue
            callback (Callable): a callback that can access the intermediate data
                                 during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion (bool): an automatic conversion for operator and aux_operators into
                                    the type which is
                                    most suitable for the backend.
                                    - non-aer statevector_simulator: MatrixOperator
                                    - aer statevector_simulator: WeightedPauliOperator
                                    - qasm simulator or real backend:
                                        TPBGroupedWeightedPauliOperator
        """
        self.validate(locals())
        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point)
        self._use_simulator_snapshot_mode = None
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        if initial_point is None:
            self._initial_point = var_form.preferred_init_points
        self._operator = operator
        self._eval_count = 0
        self._aux_operators = []
        if aux_operators is not None:
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            for aux_op in aux_operators:
                self._aux_operators.append(aux_op)
        self._auto_conversion = auto_conversion
        logger.info(self.print_settings())
        self._var_form_params = ParameterVector('θ', self._var_form.num_parameters)

        self._parameterized_circuits = None
###############################################################################################################################################################
[docs]    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance

        Returns:
            VQE: vqe object
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        vqe_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        initial_point = vqe_params.get('initial_point')
        max_evals_grouped = vqe_params.get('max_evals_grouped')

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, var_form, optimizer,
                   initial_point=initial_point, max_evals_grouped=max_evals_grouped,
                   aux_operators=algo_input.aux_ops)


    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

[docs]    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.configuration['name'])
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._var_form.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

#####################################################################################################################################################################
    def _config_the_best_mode(self, operator, backend): #returns the given operator that is rewritten into a format that can be most quickly calculated on the backend simulator

        if not isinstance(operator, (WeightedPauliOperator, MatrixOperator, #throw error if passed wrong operator class type
                                     TPBGroupedWeightedPauliOperator)):
            logger.debug("Unrecognized operator type, skip auto conversion.")
            return operator

        ret_op = operator
        if not is_statevector_backend(backend) and not ( #if using QASM simulator
                is_aer_provider(backend)
                and self._quantum_instance.run_config.shots == 1):
            if isinstance(operator, (WeightedPauliOperator, MatrixOperator)):
                logger.debug("When running with Qasm simulator, grouped pauli can "
                             "save number of measurements. "
                             "We convert the operator into grouped ones.")
                ret_op = op_converter.to_tpb_grouped_weighted_pauli_operator( #convert operator into pauli string?
                    operator, TPBGroupedWeightedPauliOperator.sorted_grouping)
        else:
            if not is_aer_provider(backend): #if using non-aer statevector simulator
                if not isinstance(operator, MatrixOperator): 
                    logger.info("When running with non-Aer statevector simulator, "
                                "represent operator as a matrix could "
                                "achieve the better performance. We convert "
                                "the operator to matrix.")
                    ret_op = op_converter.to_matrix_operator(operator) #convert operator into matrix
            else:
                if not isinstance(operator, WeightedPauliOperator): #if using aer simulator
                    logger.info("When running with Aer simulator, "
                                "represent operator as weighted paulis could "
                                "achieve the better performance. We convert "
                                "the operator to weighted paulis.")
                    ret_op = op_converter.to_weighted_pauli_operator(operator) #convert operator into weighted pauli sum
        return ret_op
#################################################################################################################################################################
[docs]    def construct_circuit(self, parameter, statevector_mode=False,
                          use_simulator_snapshot_mode=False, circuit_name_prefix=''): #is this supposed to be defined within _config_the _best_mode?
        """Generate the circuits.

        Args:
            parameter (numpy.ndarray): parameters for variational form.
            statevector_mode (bool, optional): indicate which type of simulator are going to use.
            use_simulator_snapshot_mode (bool, optional): is backend from AerProvider,
                            if True and mode is paulis, single circuit is generated.
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            list[QuantumCircuit]: the generated circuits with Hamiltonian. 
        """ #returns a list of quantum circuits that measure each pauli string in the 
        wave_function = self._var_form.construct_circuit(parameter) #creates QuantumCircuit that can implement parameter-based evolutions on our initial state (passed on creation of var form)
        circuits = self._operator.construct_evaluation_circuit( #creates circuit that measures each pauli string in operator using the given wavefunction
            wave_function, statevector_mode, #the self._operator is a parameter passed upon initial creation of VQE object
            use_simulator_snapshot_mode=use_simulator_snapshot_mode,
            circuit_name_prefix=circuit_name_prefix)
        return circuits 


    def _eval_aux_ops(self, threshold=1e-12, params=None): #essentially this finds expectation values for operators for our ansatz above certain threshold
        if params is None:
            params = self.optimal_params
        wavefn_circuit = self._var_form.construct_circuit(params)
        circuits = []
        values = []
        params = []
        for idx, operator in enumerate(self._aux_operators):
            if not operator.is_empty():
                temp_circuit = QuantumCircuit() + wavefn_circuit #why do we add QuantumCircuit?
                circuit = operator.construct_evaluation_circuit(
                    wave_function=temp_circuit,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                    circuit_name_prefix=str(idx))
            else:
                circuit = None
            circuits.append(circuit)

        if circuits:
            to_be_simulated_circuits = \
                functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
            result = self._quantum_instance.execute(to_be_simulated_circuits)

            for idx, operator in enumerate(self._aux_operators):
                if operator.is_empty():
                    mean, std = 0.0, 0.0
                else:
                    mean, std = operator.evaluate_with_result(
                        result=result, statevector_mode=self._quantum_instance.is_statevector,
                        use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                        circuit_name_prefix=str(idx))

                mean = mean.real if abs(mean.real) > threshold else 0.0
                std = std.real if abs(std.real) > threshold else 0.0
                values.append((mean, std))

        if values:
            aux_op_vals = np.empty([1, len(self._aux_operators), 2])
            aux_op_vals[0, :] = np.asarray(values)
            self._ret['aux_ops'] = aux_op_vals
###################################################################################################################################################################
    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        if self._auto_conversion: #is set to True by default upon initialization
            self._operator = \
                self._config_the_best_mode(self._operator, self._quantum_instance.backend) #convert operator to best operator format for backend
            for i in range(len(self._aux_operators)):
                if not self._aux_operators[i].is_empty():
                    self._aux_operators[i] = \
                        self._config_the_best_mode(self._aux_operators[i],
                                                   self._quantum_instance.backend) #convert aux operators into optimal format for backend

        # sanity check
        if isinstance(self._operator, MatrixOperator) and not self._quantum_instance.is_statevector:
            raise AquaError("Non-statevector simulator can not work "
                            "with `MatrixOperator`, either turn ON "
                            "auto_conversion or use the proper "
                            "combination between operator and backend.") #if you didn't autoconvert and you passed bad operator type then you might get this error

        self._use_simulator_snapshot_mode = ( #is boolean and true if you're using aer with 1 shot and no noise? and your operator is weighted pauli
            is_aer_provider(self._quantum_instance.backend)
            and self._quantum_instance.run_config.shots == 1
            and not self._quantum_instance.noise_config
            and isinstance(self._operator,
                           (WeightedPauliOperator, TPBGroupedWeightedPauliOperator)))

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0
        self._ret = self.find_minimum(initial_point=self.initial_point, #finds optimum, returns dictionary with number of evaluations, optimum cost fun value, optimum parameters, and evaluation time
                                      var_form=self.var_form,
                                      cost_fn=self._energy_evaluation,
                                      optimizer=self.optimizer) 
        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals'] #set eval count to min number of optimizer evals, though eval count should start at inf not 0 right?
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self.get_optimal_cost()]) #returns optimal energy as a single array entry, and isn't this the same as energy?
        self._ret['eigvecs'] = np.asarray([self.get_optimal_vector()]) #returns a statevector representation (0 1 basis) of the variational form ansatz
        self._eval_aux_ops()

        self.cleanup_parameterized_circuits()
        return self._ret #returns dictionary with eval count, min energy, eigenvals and eigenvects
####################################################################################################################################################################
    # This is the objective function to be passed to the optimizer that is used for evaluation, so this is cost_fn
    #but this doesn't return anything, how does it act as a function? -nvm, _build_parameterized_circuit is defined within energy_evaluation.
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            Union(float, list[float]): energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        parameter_sets = np.split(parameters, num_parameter_sets) #split passed parameters into parameter sets
        mean_energy = []
        std_energy = []
        #don't we need to set self._var_form_params = parameters here?
        def _build_parameterized_circuits(): #construct parameterized circuit based on the self parameters, no returns but defined the circuits in self._parameterized_circuits property
            if self._var_form.support_parameterized_circuit and \
                    self._parameterized_circuits is None: #does all of this only if var form is set up to support parameterized circuit and if parameterized circuit hasn't already been set up
                parameterized_circuits = self.construct_circuit( #returns list of circuits that evaluate varform wavefunction measurement of each pauli string in the VQE operator (typically Hamiltonian)
                    self._var_form_params, #these should be the arg parameters no?
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode)

                self._parameterized_circuits = \
                    self._quantum_instance.transpile(parameterized_circuits)# a list of transpiled quantum circuits that can be run on backend specificed by quantum instance
        _build_parameterized_circuits() # this runs the previously defined function within _energy_evaluation
        circuits = []
        # binding parameters here since the circuits had been transpiled # what does binding mean and why occur after transpilation?
        if self._parameterized_circuits is not None: 
            for idx, parameter in enumerate(parameter_sets): #here is where the passed parameters enter in, is the binding to make sure the passed parameters are the ones used instead of all zeros (self._var_form_params)?
                curr_param = {self._var_form_params: parameter} #but here we're still drawing from self._var_form_params?
                for qc in self._parameterized_circuits:
                    tmp = qc.bind_parameters(curr_param)
                    tmp.name = str(idx) + tmp.name
                    circuits.append(tmp)
            to_be_simulated_circuits = circuits
        else:
            for idx, parameter in enumerate(parameter_sets):
                circuit = self.construct_circuit(
                    parameter,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                    circuit_name_prefix=str(idx))
                circuits.append(circuit)
            to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)

        result = self._quantum_instance.execute(to_be_simulated_circuits,
                                                self._parameterized_circuits is not None) #execute the circuit to get result object

        for idx, _ in enumerate(parameter_sets):
            mean, std = self._operator.evaluate_with_result(
                result=result, statevector_mode=self._quantum_instance.is_statevector,
                use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                circuit_name_prefix=str(idx)) #gets mean and std of each result for each parameter set?
            mean_energy.append(np.real(mean))
            std_energy.append(np.real(std))
            self._eval_count += 1
            if self._callback is not None:
                self._callback(self._eval_count, parameter_sets[idx], np.real(mean), np.real(std))
            logger.info('Energy evaluation %s returned %s', self._eval_count, np.real(mean))

        return mean_energy if len(mean_energy) > 1 else mean_energy[0] #returns an array of the mean energies of each parameter set (for us we'll use only 1), we may want this to return std too.
############################################################################################################################################
[docs]    def get_optimal_cost(self): #returns min value from find_minimum return list
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']


[docs]    def get_optimal_circuit(self): #returns var form with parameters given by the optimal parameters from find_minimum function
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])


[docs]    def get_optimal_vector(self): 
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc) 
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector'] #returns a state vector representation of the variational form ansatz


    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret: 
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']