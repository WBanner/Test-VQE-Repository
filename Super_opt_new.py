from qiskit.aqua.components.optimizers import COBYLA, L_BFGS_B
from optimizer_new import Optimizer
from bfgs_grad_new import BFGS_Grad
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt.utils import create_result
from skopt.space import Space, Real


class SuperOptimizer:
    """Create more robust, featureful optimizers for use in Qiskit.

    To use this class, simply take an optimizer like `COBYLA` and create
    a class like
    ```
    class SuperCOBYLA(SuperOptimizer, COBYLA):
        pass
    ```

    Now you can make an optimizer like `opt = SuperCOBYLA(maxiter=50)`.
    This optimizer now allows automatic restarts by using the keyword argument
    `SuperCOBYLA(maxiter=50, _num_restarts=10)`.

    Once an optimization has been run, you can use the `opt.optimization_result`
    attribute with the plotting functions in `skopt`.

    https://scikit-optimize.github.io/stable/modules/plots.html#plots

    Note that some of the functions in `skopt.plots` require a model. This can
    be created by also supplying the appropriate keyword argument:
    ```
    opt = SuperCOBYLA(maxiter=50, _make_model=True)
    ```
    By default, this will only work if less than 200 points are found in the
    optimization for performance reasons. This can be overridden with 
    `_max_model_points`.

    Note that the model here uses Gaussian processes to approximate the rest of
    the objective function. Since an optimizer may not (and generally does not)
    sample lots of parameter space, one should only trust the model to make accurate
    predictions about the objective function near/between points that it has
    directly sampled.
    """

    def __init__(self, *args, **kwargs):
        self.optimization_result = None
        self._num_restarts = kwargs.pop('_num_restarts', None)
        self._make_model = kwargs.pop('_make_model', False)
        self._max_model_points = kwargs.pop('_max_model_points', 200)
        super().__init__(*args, **kwargs)
    
    def optimize(
        self, 
        num_vars, 
        objective_function, 
        gradient_function=None,
        variable_bounds=None, 
        initial_point=None
        ):

        callbacks = []
        
        def alt_obj_fn(pars):
            fn = objective_function(pars)
            callbacks.append({
                'point': pars,
                'fn': fn
            })
            return fn
        
        result = super().optimize(
            num_vars,
            alt_obj_fn,
            gradient_function,
            variable_bounds,
            initial_point
        )

        if self._num_restarts is not None:
            for i in range(self._num_restarts-1):
                if variable_bounds is not None:
                    init_pt = [
                        np.random.uniform(dn, up)
                        for dn, up in variable_bounds
                    ]
                else:
                    init_pt = [
                        np.random.uniform(-np.pi, +np.pi)
                        for _ in range(num_vars)
                    ]
                result_new = super().optimize(
                    num_vars,
                    alt_obj_fn,
                    gradient_function,
                    variable_bounds,
                    init_pt
                    )
                if result_new[1] < result[1]:
                    result = result_new

        X = [step['point'] for step in callbacks]
        y = [step['fn'] for step in callbacks]
        if self._make_model and (len(callbacks) < self._max_model_points):
            model = GaussianProcessRegressor()
            model.fit(X, y)
        else:
            model = None
        
        if variable_bounds is not None:
            space = Space([
                Real(low, high)
                for low, high in variable_bounds
            ])
        else:
            space = None
        
        self.optimization_result = create_result(
            X, 
            y, 
            space=space,
            models=[model] if model is not None else None
            )
        
        return result

class SuperCOBYLA(SuperOptimizer, COBYLA):
    pass


if __name__ == '__main__':
    from scipy.optimize import rosen

    sup_cob = SuperCOBYLA(_num_restarts=1, _make_model=True, maxiter=50)

    result = sup_cob.optimize(3, rosen, initial_point=[0.0, 0.1, 0.2])

    print(sup_cob.optimization_result)
    print(result)

class SuperBFGS_Grad(SuperOptimizer, BFGS_Grad):
    pass

class SuperL_BFGS_B(SuperOptimizer, L_BFGS_B):
    pass