"""Rotosolve is an optimization algorithm used for quantum variational algorithms.
It leverages the structure of varitional forms and trigonometric identities to
reduce the number of circuits executed on the quantum device.
Also see the `qisresearch.i_vqe.rotoselect.RotoSelect` algorithm.
Quantum circuit structure learning
Mateusz Ostaszewski, Edward Grant, Marcello Benedetti
https://arxiv.org/abs/1905.09692
"""
from qiskit.aqua.components.optimizers import Optimizer
import numpy as np

class Rotosolve(Optimizer):
    """Create an instance of the Rotosolve optimizer.
    Parameters
    ----------
    max_steps : int
        Maximum number of steps to take in the optimizer. This is the number
        of times to loop through all the parameters in the objective function.
    alt_convention : bool
        The Rotosolve paper (see above) uses the convention that there is a `1/2`
        in the exponent. This convention corresponds to `alt_convention=False`.
        In other algorithms (e.g. `qisresearch.i_vqe.adapt.ADAPTVQE`), this factor is
        not present, in which case, use `alt_convention=True`.
    """

    CONFIGURATION = {
        'name': 'RotoSolve',
        'description': 'RotoSolve Optimizer',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'GP_schema',
            'type': 'object',
            'properties': {},
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.supported
        },
        'options': [],
        'optimizer': []
    }

    def __init__(
        self,
        max_steps: int,
        alt_convention: bool = False,
        param_per_step: int = 1 #will added
    ):
        super().__init__()
        self._max_steps = max_steps
        self.param_per_step = param_per_step #will added

        # alt_convention == True:
        # e^{-i H theta}
        # alt_convention == False
        # e^{-i H theta/2}
        self._alt_convention = alt_convention

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        """See `qiskit.aqua.components.optimizers.Optimizer` documentation.
        """
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        factor = 2 if self._alt_convention else 1

        def f(x):
            return objective_function(np.array(x) / factor)

        if initial_point is None:
            initial_point = np.random.uniform(-np.pi,+np.pi, num_vars)

        res = self._rotosolve(
            f,
            initial_point,
            self._max_steps,
            self.param_per_step #will added
        )

        self.res = res

        return res

    @staticmethod
    def _rotosolve(f, initial_point: np.array, max_steps: int, param_per_step = 1): #will added
        D = len(initial_point)
        theta = initial_point

        f_evals = 0
        def f_counter(*args, **kwargs):
            return f(*args, **kwargs)

        f_current = f_counter(initial_point)
        f_evals += 1
        converged = False
        steps = 0

        theta_values = []
        f_values = []

        if D < param_per_step: #will added (need to replace with actual error)
            print('error')

        if param_per_step == 1:#will added
            while not converged:
                for d in range(D):
                    phi = np.random.uniform(-np.pi, +np.pi)
                    theta_d = phi
                    theta[d] = theta_d
                    MVals = {
                        'phi+0': 0,
                        'phi+pi/2': 0,
                        'phi-pi/2': 0
                    }

                    MVals['phi+0'] = f_counter(theta)
                    f_evals += 1
                    theta[d] = theta[d] + np.pi/2
                    MVals['phi+pi/2'] = f_counter(theta)
                    f_evals += 1
                    theta[d] = theta[d] - np.pi
                    MVals['phi-pi/2'] = f_counter(theta)
                    f_evals += 1

                    theta[d] = phi - np.pi/2 - np.arctan2(
                        2*MVals['phi+0'] - MVals['phi+pi/2'] - MVals['phi-pi/2'],
                        MVals['phi+pi/2'] - MVals['phi-pi/2']
                    )

                    phi = 0
                    theta_d = 0

                theta_values.append(theta)
                f_values.append(f(theta))

                steps += 1
                if steps >= max_steps:
                    converged = True

        if param_per_step == 2: #will added
            while not converged:
                for d in range(D - 1):
                    theta[d] = 0
                    theta[d+1] = -np.pi/2
                    MVals = {
                        '0': [0,0,0],
                        'pi/2': [0,0,0],
                        '-pi/2':[0,0,0]
                    }
                    A = np.empty(0)
                    B = np.empty(0)
                    C = np.empty(0)

                    for num,i in enumerate([-np.pi/2, 0, np.pi/2]):
                        theta[d] = i
                        MVals['-pi/2'][num] = f_counter(theta)
                        f_evals += 1
                    theta[d+1] = np.pi/2
                    theta[d] = 0
                    for num,i in enumerate([-np.pi/2, 0, np.pi/2]):
                        theta[d] = i
                        MVals['pi/2'][num] = f_counter(theta)
                        f_evals += 1
                    theta[d+1] = 0
                    theta[d] = 0
                    for num,i in enumerate([-np.pi/2, 0, np.pi/2]):
                        theta[d] = i
                        MVals['0'][num] = f_counter(theta)
                        f_evals += 1

                    B = np.arctan2(-np.array(MVals['pi/2']) - np.array(MVals['-pi/2']) + 2*np.array(MVals['0']),np.array(MVals['pi/2'])- np.array(MVals['-pi/2']))
                    X = np.sin(B)
                    Y = np.sin(B + np.pi/2)
                    Z = np.sin(B - np.pi/2)
                    for i in range(0,(len(MVals['pi/2']))):
                        if Y[i] != 0:
                            C = np.append(C, (MVals['0'][i]-MVals['pi/2'][i]*(X[i]/Y[i]))/(1-X[i]/Y[i]))
                            A = np.append(A, (MVals['pi/2'][i] - C[-1])/Y[i])
                        else:
                            C = np.append(C, MVals['pi/2'][i])
                            A = np.append(A, (MVals['0'][i] - C[-1])/X[i])
                    sins = A[0]*np.sin(B[0])+A[1]*np.sin(B[1])
                    coss = A[0]*np.cos(B[0])+A[1]*np.cos(B[1])
                    sinarg = (-C[0]-C[1])/np.sqrt(np.square(coss) + np.square(sins))
                    theta[d+1] = np.arcsin(sinarg) - np.arctan2(sins,coss)
                    K_1 = A[0]*np.cos(theta[d+1]+B[0])
                    K_3 = A[1]*np.cos(theta[d+1]+B[1])
                    K_2 = A[2]*np.cos(theta[d+1]+B[2])
                    coss = np.sqrt(2)*K_3/4 - np.sqrt(2)*K_1/4
                    sins = np.sqrt(2)*K_3/4 + np.sqrt(2)*K_1/4
                    tans = np.arctan2(sins,coss)
                    L = np.sqrt(np.square(coss) + np.square(sins))
                    M = np.sqrt(1+L+2*L*np.cos(tans))
                    tans_2 = np.tan(L*np.sin(tans)/(L*np.sin(tans)+1))
                    theta[d] = np.arccos(-(K_3+K_1)/M) - tans_2
                    #will added


                theta_values.append(theta)
                f_values.append(f(theta))

                steps += 1
                if steps >= max_steps:
                    converged = True

        f_current = f_counter(theta)
        f_evals += 1

        min_index = np.argmin(f_values)
        f_min = f_values[min_index]
        theta_min = theta_values[min_index]

        return theta_min, f_min, f_evals