#!/usr/bin/env python
# coding: utf-8

# # RotoSolve

# https://arxiv.org/pdf/1905.09692.pdf

# In[1]:


from qiskit.aqua.components.optimizers import Optimizer
import numpy as np


# In[75]:


class Rotosolve(Optimizer):
    
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
        min_energy_change = 1e-4,
        max_steps = 5
    ):
        super().__init__()
        self._max_steps = max_steps
        self.min_energy_change = min_energy_change
    
    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        
        def f(x):
            return objective_function(np.array(x))
        
        if initial_point is None:
            initial_point = np.random.uniform(-np.pi,+np.pi, num_vars)
        
        res = self._rotosolve(
            f,
            initial_point,
            self._max_steps,
            self.min_energy_change
        )
        
        self.res = res
        
        return res
    
    @staticmethod
    def _rotosolve(f, initial_point: np.array, max_steps: int, min_energy_change):
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
        
        while not converged:
            for d in range(D):
                phi = 0 #np.random.uniform(-np.pi, +np.pi) can do this later
                theta_d = phi
                theta[d] = theta_d
                MVals = {
                    'phi+0': 0,
                    'phi+pi/2': 0,
                    'phi-pi/2': 0
                }
                
                MVals['phi+0'] = f_counter(theta)
                f_evals += 1
                theta[d] = theta[d] + np.pi/4 #forgot to divide by 4 instead of 2
                MVals['phi+pi/2'] = f_counter(theta)
                f_evals += 1
                theta[d] = theta[d] - np.pi/2 #forgot to divide by 2 instead of 0
                MVals['phi-pi/2'] = f_counter(theta)
                f_evals += 1
                
                theta[d] = phi - np.pi/4 - np.arctan2(
                    2*MVals['phi+0'] - MVals['phi+pi/2'] - MVals['phi-pi/2'],
                    MVals['phi+pi/2'] - MVals['phi-pi/2']
                )/2 #forgot to divide all by2
                
                phi = 0
                theta_d = 0
            
            theta_values.append(theta)
            f_values.append(f(theta))
            
            steps += 1
            if len(f_values) >=2  and abs(f_values[-1] - f_values[-2]) < min_energy_change:
                converged = True
            if steps >= max_steps:
                converged = True
        
        f_current = f_counter(theta)
        f_evals += 1
        
        min_index = np.argmin(f_values)
        f_min = f_values[min_index]
        theta_min = theta_values[min_index]
        
        return theta_min, f_min, f_evals


