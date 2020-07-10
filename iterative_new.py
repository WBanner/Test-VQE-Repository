"""`IterativeVQE` is a generic algorithm that runs a sequence of VQEs, feeding
the input of each into the next based on some rules. By sub-classing this class,
one is able to quickly implement a variety of quantum algorithms.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

from typing import Dict, List
from abc import abstractmethod
from qiskit.aqua import AquaError
from copy import deepcopy
from qiskit.aqua.algorithms import QuantumAlgorithm, VQE


class IterativeVQE(QuantumAlgorithm):
    """Base class for all sub-classes of iterative VQE algorithms.

    Parameters
    ----------
    return_best_result : bool
        Whether or not to return the result corresponding to the lowest energy.

    Attributes
    ----------
    CONFIGURATION : dict
        Configuration of the algorithm for `qiskit.aqua` support.
    step_history: List[Dict]
        A list of all of the results produced by the VQE at each step. May also
        include additional details depending on the specific algorithm. This
        attribute is the input to several plotting functions under
        `qisresearch.i_vqe.vis`.
    last_result: Dict
        The result produced by the last VQE run.

    """

    CONFIGURATION = {
        'name': 'IterativeVQE',
        'description': 'Iterative VQE',
    }

    def __init__(self, return_best_result: bool = False):
        self.return_best_result = return_best_result
        super().__init__()

    @property
    def step_history(self) -> List[Dict]:
        try:
            self._step_history
        except AttributeError:
            self._step_history = []
        return self._step_history

    def update_step_history(self, result: Dict):
        try:
            self._step_history
        except AttributeError:
            self._step_history = []
        self._step_history.append(result)
        logger.info(
            'Finished iVQE step {} with result: {}'.format(
                self.step,
                result
            )
        )
        try:
            logger.info('iVQE step {} reported energy {}'.format(
                self.step,
                result['energy']
            ))
            logger.info('iVQE step {} reported parameters {}'.format(
                self.step,
                result['opt_params']
            ))
        except KeyError:
            pass
        self._step += 1

    @property
    def last_result(self) -> Dict:
        return self._step_history[-1]


    @property
    def step(self) -> int:
        try:
            self._step
        except AttributeError:
            self._step = 0
        return self._step

    def is_converged(self) -> bool:
        base_conv = self._is_converged()
        try:
            self.step_callbacks
        except:
            return base_conv
        callback_results = [
            callback._halt(self.step_history)
            for callback in self.step_callbacks
        ]
        for res, callback in zip(callback_results, self.step_callbacks):
            if res:
                self._step_history[-1]['termination'] = callback.halt_reason(self.step_history)
                return True
        if base_conv:
            self._step_history[-1]['termination'] = 'Base convergence'
        return base_conv

    @abstractmethod
    def _is_converged(self) -> bool:
        pass

    @abstractmethod
    def first_vqe_kwargs(self) -> Dict:
        pass

    @abstractmethod
    def next_vqe_kwargs(self, last_result) -> Dict:
        pass

    def post_process_result(self, result, vqe, last_result) -> Dict:
        result['current_circuit'] = vqe.get_optimal_circuit()
        sv = self._quantum_instance.is_statevector
        if sv:
            result['std_devs'] = False
            result['energy_std'] = 0.0
        else:
            try:
                result['std_devs'] = True
                eval_circs = vqe._operator.construct_evaluation_circuit(result['current_circuit'], sv)
                eval_result = self._quantum_instance.execute(eval_circs)
                mean, std = vqe._operator.evaluate_with_result(result=eval_result, statevector_mode=sv)
                result['energy_std'] = std.real
            except:
                result['std_devs'] = False
                result['energy_std'] = 0.0
        return result

    def _first_iteration(self) -> Dict:
        logger.info('Starting iVQE step {}'.format(self.step))
        vqe = VQE(**self.first_vqe_kwargs()) #will edited
        result = deepcopy(vqe.run(self.quantum_instance))
        result = self.post_process_result(result, vqe, None)
        result['optimizer_counter'] = 1
        logger.info('Finished iVQE step {}'.format(self.step))
        del vqe
        self.update_step_history(result)
        return 0

    def _iteration(self, last_result: Dict) -> Dict:
        logger.info('Starting iVQE step {}'.format(self.step))
        vqe = VQE(**self.next_vqe_kwargs(last_result))
        result = {'energy': 10000}
        counter = 0
        while result['energy'] > last_result['energy'] + 1.1e-10:
            result = vqe.run(self.quantum_instance)
            counter = counter + 1
        result['optimizer_counter'] = counter
        result = self.post_process_result(result, vqe, last_result)
        logger.info('Finished iVQE step {}'.format(self.step))
        self.update_step_history(result)
        del vqe
        return 0

    def _run(self) -> Dict:
        counter = 0
        self._first_iteration()
        while not self.is_converged():
            print('finished', counter)
            self._iteration(self.last_result)
            counter = counter + 1
        logger.info('Finished final iVQE step {}'.format(self.step))
        if not self.return_best_result:
            self._optimal_circuit = self.last_result['current_circuit']
            return self._step_history #will edited
        else:
            energies = np.array([step['energy'] for step in self.step_history])
            min_ind = np.argmin(energies)
            self.best_result = self.step_history[min_ind]
            self._optimal_circuit = self.best_result['current_circuit']
            return self._step_history #will edited

    def get_optimal_circuit(self):
        if self._optimal_circuit is None:
            raise AquaError(
                "Cannot find optimal circuit before running the algorithm to find optimal params.")
        return self._optimal_circuit