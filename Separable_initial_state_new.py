import logging

from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.initial_states import InitialState

from adapt_var_form import SingleQubitRotations,SingleQubitRotationsReal

logger = logging.getLogger(__name__)


class SeparableInitialState(InitialState):
    """An initial state constructed from a completely separable VQE ansatz."""

    CONFIGURATION = {
        'name': 'VQE-Separable-State',
        'description': 'VQE Separable initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'separable_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(
            self,
            operator,
            optimizer,
            **vqe_kwargs
    ):
        """Initialize separable state object.
        Parameters
        ----------
        operator : Operator
            Operator to minimize the separable state with respect to.
        optimizer : Optimzer
            Optimizer to use for determining optimal separable state.
        **vqe_kwargs : type
            Options for VQE that determines optimal initial state.
        """
        super().__init__()
        self.vqe = VQE(
            operator=operator,
            var_form=SingleQubitRotations(operator.num_qubits),
            optimizer=optimizer,
            **vqe_kwargs
        )
        self._circuit = None
        self._result = None

    def initialize(self, quantum_instance):
        """Initialize the separable initial state.
        Parameters
        ----------
        quantum_instance : QuantumInstance
            What device/simulator to use for determining the optimal separable state.
        """
        if self._circuit is None:
            self._result = self.vqe.run(quantum_instance)
            logger.info('Found initial separable state cost {}'.format(self.vqe.get_optimal_cost()))
            self._circuit = self.vqe.get_optimal_circuit()
            return self._circuit
        else:
            return self._circuit

    def construct_circuit(self, mode='circuit', register=None):
        """Return the optimal separable state.
        Parameters
        ----------
        mode : str
            Mode to use for construction.
        register : type
            Register to use for constructing circuit. Currently unused.
        Returns
        -------
        QuantumCircuit
            Circuit that prepares optimal separable state.
        """
        if mode != 'circuit':
            raise ValueError('Selected mode {} is not supported'.format(mode))
        if self._circuit is None:
            raise ValueError('Initial state has not yet been initialized.')
        circuit = self._circuit
        return circuit

class SeparableInitialStateReal(InitialState):
    """An initial state constructed from a completely separable VQE ansatz."""

    CONFIGURATION = {
        'name': 'VQE-Separable-State',
        'description': 'VQE Separable initial state',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'separable_state_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        }
    }

    def __init__(
            self,
            operator,
            optimizer,
            **vqe_kwargs
    ):
        """Initialize separable state object.
        Parameters
        ----------
        operator : Operator
            Operator to minimize the separable state with respect to.
        optimizer : Optimzer
            Optimizer to use for determining optimal separable state.
        **vqe_kwargs : type
            Options for VQE that determines optimal initial state.
        """
        super().__init__()
        self.vqe = VQE(
            operator=operator,
            var_form=SingleQubitRotationsReal(operator.num_qubits),
            optimizer=optimizer,
            **vqe_kwargs
        )
        self._circuit = None
        self._result = None

    def initialize(self, quantum_instance):
        """Initialize the separable initial state.
        Parameters
        ----------
        quantum_instance : QuantumInstance
            What device/simulator to use for determining the optimal separable state.
        """
        if self._circuit is None:
            self._result = self.vqe.run(quantum_instance)
            logger.info('Found initial separable state cost {}'.format(self.vqe.get_optimal_cost()))
            self._circuit = self.vqe.get_optimal_circuit()
            return self._circuit
        else:
            return self._circuit

    def construct_circuit(self, mode='circuit', register=None):
        """Return the optimal separable state.
        Parameters
        ----------
        mode : str
            Mode to use for construction.
        register : type
            Register to use for constructing circuit. Currently unused.
        Returns
        -------
        QuantumCircuit
            Circuit that prepares optimal separable state.
        """
        if mode != 'circuit':
            raise ValueError('Selected mode {} is not supported'.format(mode))
        if self._circuit is None:
            raise ValueError('Initial state has not yet been initialized.')
        circuit = self._circuit
        return circuit