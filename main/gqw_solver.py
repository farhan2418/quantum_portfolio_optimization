r"""
This module has the guided quantum walk -algorithm developed by sebastian schulz et. al solver
which optimally selects/tunes the lambda vector to get the parameteres for the hamiltonian simulation.
The class structure follows what qiskit has done for other variational algorithms like SamplingVQE to adapat with their optimizers 
"""

from typing import Callable, Any
import numpy as np
import random
import logging

from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import BaseSamplerV2
from qiskit.result import QuasiDistribution


from qiskit_algorithms.utils.validation import validate_min
from qiskit_algorithms.optimizers import Minimizer, Optimizer, OptimizerResult
from qiskit_algorithms.minimum_eigensolvers.sampling_vqe import _DiagonalEstimator
from qiskit_algorithms.minimum_eigensolvers.sampling_mes import (
    SamplingMinimumEigensolver,
    SamplingMinimumEigensolverResult,
)
from qiskit_algorithms.variational_algorithm import (
    VariationalAlgorithm, VariationalResult
)
from gqw_ansatz import GQWAnsatz
from helper import (compare_measurements, calc_gamma)

logger = logging.getLogger(__name__)

class SamplingGQWResult(VariationalResult, SamplingMinimumEigensolverResult):
    """The Guided Variational Quantum walk Result."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_function_evals: int | None = None

    @property
    def cost_function_evals(self) -> int | None:

        return self._cost_function_evals

    @cost_function_evals.setter
    def cost_function_evals(self, value: int) -> None:

        self._cost_function_evals = value

class GQW(VariationalAlgorithm, SamplingMinimumEigensolver):
    def __init__(self,
    sampler: BaseSamplerV2,
    optimizer: Optimizer | Minimizer,
    *,
    reps: int = 1,
    time: int = 1,
    initial_state: QuantumCircuit | None = None,
    driver: QuantumCircuit | BaseOperator = None,
    initial_lambda: np.ndarray | None = None, #this can be STD_ARRAY,
    callback: Callable[[int, np.ndarray, float, dict[str, Any]], None] | None = None,
    ) -> None:
        
        validate_min("reps", reps, 1)

        self.sampler = sampler
        self.optimizer = optimizer
        self.reps = reps
        self.driver = driver
        self.initial_state = initial_state
        self.ansatz = None
        self.callback = callback
        self.time = time
        
        # getter and setter for initial point for lambda vector that needs to be optimized
        self._initial_lambda = initial_lambda
        self._cost_operator = None

    @property
    def initial_point(self) -> np.ndarray | None:
        """Return the initial point."""
        return self._initial_lambda

    @initial_point.setter
    def initial_point(self, value: np.ndarray | None) -> None:
        """Set the initial point."""
        self._initial_lambda = value

    @classmethod
    def supports_aux_operators(cls) -> bool:
        return True

    def _assign_ansatz(self, operator: BaseOperator):
        ansatz = GQWAnsatz(
            operator, reps= self.reps, initial_state= self.initial_state, driver= self.driver, flatten= False
        )

        self.ansatz = ansatz

        
    
    def _validate_initial_lambda(self, initial_point: np.ndarray| None):

        if initial_point is None:
            lower_bound = 0
            upper_bound = 1
            initial_point =  [round(random.uniform(lower_bound, upper_bound), 3) for _ in range(4)] +\
                  [round(random.random(), 3) for _ in range(2)]
        if len(initial_point) != 6 :
            raise ValueError("The tuning lambda vector must be of length 6")
        
        # assuming the passed lambda vectors are in order   
        inclusie_points = initial_point[0: 4]
        exclusive_points = initial_point[4:6]
        for i, l in enumerate(inclusie_points):
            if not (0.0 <= l <= 1.0):
                raise ValueError(
                    "lambda points l1, l2, l3, l4 shold be in limit [0, 1]",
                    f"but {l} val of index {i} is not "
                )
            
        for l in exclusive_points:
            if not (0.0 < l < 1.0):
                raise ValueError(
                    "lambda points l5, l6 should be in limit (0, 1)",
                    f"but {l} is not"
                )
        return initial_point

    def _get_evaluate_energy(
            self,
            operator: BaseOperator,
            ansatz: QuantumCircuit,
            return_best_measurement: bool = False,
    ) -> (Callable[[np.ndarray], np.ndarray | float ] 
                  | tuple[Callable[[np.ndarray], np.ndarray | float], dict[str, Any]] 
        ):
        num_parameters = ansatz.num_parameters
        if num_parameters == 0:
            raise ValueError("The ansatz must be parameterized, but has 0 parameters")
        
        eval_count = 0

        best_measurement = {"best": None}
        
        def store_best_measurement(best):
            for best_i in best:
                if best_measurement["best"] is None or compare_measurements(
                    best_i, best_measurement["best"]
                ):
                    best_measurement["best"] = best_i

        estimator = _DiagonalEstimator(
            sampler = self.sampler,
            callback = store_best_measurement 
        )
        def evaluate_energy(lambda_parameters: np.ndarray) -> np.ndarray | float:
            nonlocal eval_count
            delta_T = self.time / self.reps

            parameters = [delta_T *
            calc_gamma( i * delta_T, self.time, lambda_parameters) for i in range (self.reps)
            ] + [delta_T for _ in range(self.reps)]
            parameters = np.reshape(parameters, (-1, num_parameters)).tolist()

            job = estimator.run([(ansatz, operator, parameters)])

            estimator_result = job.result()[0]
            values = estimator_result.data.evs

            if not values.shape:
                values = values.reshape(1)
            
            if self.callback is not None:
                for params, value in zip(parameters, values):
                    eval_count += 1
                    self.callback(eval_count, params, value, estimator_result.metadata)

            result = values if len(values) > 1 else values[0]
            return np.real(result)

        if return_best_measurement:
            return evaluate_energy, best_measurement

        return evaluate_energy


    def compute_minimum_eigenvalue(
            self, 
            operator: BaseOperator
    ):
        self._assign_ansatz(operator)

        if len(self.ansatz.clbits) > 0:
            self.ansatz.remove_final_measurements()

        self.ansatz.measure_all()

        initial_lambda = self._validate_initial_lambda(self.initial_point)

        evaluate_energy, best_measurement = self._get_evaluate_energy(  # type: ignore[misc]
            operator, self.ansatz, return_best_measurement=True
        )

        start_time = time()

        if callable(self.optimizer):
            optimizer_result = self.optimizer(
                fun = evaluate_energy,
                x0 = initial_lambda,
                jac = None,
            )

        else:
            optimizer_result = self.optimizer.minimize(
                fun = evaluate_energy,
                x0 = initial_lambda,
            )
        
        optimizer_time = time() - start_time
        # optimizer doesn't have any bounds on the lambda vectors. need to code for this
        logger.info(
            "Optimization complete in %s seconds.\nFound opt_params %s.",
            optimizer_time,
            optimizer_result.x,
        )

        delta_T = self.time / self.reps
        optimal_params  = [delta_T *
            calc_gamma( i * delta_T, self.time, optimizer_result.x) for i in range (self.reps)
            ] + [delta_T for _ in range(self.reps)] 

        final_res = self.sampler.run([(self.ansatz, optimal_params)]).result()
        final_state = getattr(final_res[0].data, self.ansatz.cregs[0].name)

        final_state = {
            label : value / final_state.num_shots
            for label, value in final_state.get_counts().items()
        }

        return self._build_sampling_quantum_walk_result(
            self.ansatz.copy(),
            optimizer_result,
            best_measurement,
            final_state,
            optimizer_time
        )

    def _build_sampling_quantum_walk_result(
            self,
            ansatz: QuantumCircuit,
            optimizer_result: OptimizerResult,
            best_measurement: dict[str, Any],
            final_state: QuasiDistribution,
            optimizer_time: float,

        ) -> SamplingGQWResult:
            result = SamplingGQWResult()
            result.eigenvalue = optimizer_result.fun
            result.cost_function_evals = optimizer_result.nfev
            result.optimal_point = optimizer_result.x 
            result.optimal_parameters = dict(
                zip(self.ansatz.parameters, optimizer_result.x) 
            )
            result.optimal_value = optimizer_result.fun
            result.optimizer_time = optimizer_time
            result.optimizer_result = optimizer_result
            result.best_measurement = best_measurement["best"]
            result.eigenstate = final_state
            result.optimal_circuit = ansatz
            return result
    

