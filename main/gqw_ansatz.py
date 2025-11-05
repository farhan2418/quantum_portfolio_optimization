import numpy as np
import qiskit.quantum_info

from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import Pauli, SparsePauliOp, SparseObservable
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter

class GQWAnsatz:
    def __init__(
        self,
        cost_operator: (qiskit.quantum_info.Pauli
            | SparsePauliOp
            | SparseObservable
            | list[qiskit.quantum_info.Pauli | SparsePauliOp | SparseObservable]
        ),
        hopping_rate: list[float] | float | None = None, 
        reps: int = 1,
        time: float = 1.0,
        driver = None,
        initial_state: QuantumCircuit | None = None,
        name: str = 'GQW',
        flatten: bool | None = None,
        evolution = None
    ):

        self._cost_operator = cost_operator
        self._reps = reps
        self._time = time
        self._driver = driver
        self._initial_state = initial_state
        self._hopping_rate = hopping_rate
        self._flatten = flatten
        self._evolution = None
        self._name = name
    
    @property
    def cost_operator(self):
        return self._cost_operator
    
    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        self._cost_operator = cost_operator
        
    @property
    def reps(self) -> int:
        return self._reps
    
    @reps.setter
    def reps(self, reps:int) -> None:
        self._reps = reps
    
    @property
    def time(self) -> int:
        return self._time
    
    @time.setter
    def time(self, time:int) -> None:
        self._time = time
    
    @property
    def flatten(self) -> bool:
        return self._flatten
    
    @flatten.setter
    def flatten(self, flatten:bool) -> None:
        self._flatten = flatten

    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, name:str) -> None:
        self._name = name
    
    @property
    def initial_state(self) -> QuantumCircuit | None:
        if self._initial_state is not None:
            return self._initial_state
        
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state
        
        return None
    
    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        self._initial_state = initial_state


    @property
    def driver(self):

        if self._driver is not None:
            return self._driver

        # we only need to create a driver operator for the angles provided or number of qubits available with the cost operator

        if self.num_qubits is not None:
            num_qubits = self.num_qubits
            hypercube_terms = [
                ("I" * left + "X" + "I" * (num_qubits - left -1 ), 1) for left in range(num_qubits)
            ]
            hypercube = SparsePauliOp.from_list(hypercube_terms)
            return hypercube
    
        return None
    
    @driver.setter
    def driver(self, driver_operator) -> None:

        self._driver = driver_operator

    @property
    def operators(self) -> list:
        return [self.driver, self.cost_operator]

    @property
    def num_qubits(self) -> int:
        return self._cost_operator.num_qubits
    
   
    def _build(self) :

        if self.num_qubits == 0:
            return
        circuit = QuantumCircuit(self.num_qubits, name = self.name)
        if self.initial_state:
            circuit.compose(self.initial_state.copy(), inplace= True, copy = False)
        delta_t = self.time / self.reps
        circuits = []
        
        def _is_driver(operator: SparsePauliOp) -> bool:
            # print(operator)
            
            pauli = operator.paulis[0]
            if isinstance(pauli, Pauli):
                return np.any(pauli.x)
            
            return False
    
        def _evolve_operator(operator: SparsePauliOp, time):
            print(time)
            # print(operator)
            evolution = LieTrotter() if self._evolution is None else self._evolution
            gate = PauliEvolutionGate(operator, time, synthesis=evolution)
            evolved = QuantumCircuit(operator.num_qubits)
            if not self.flatten:
                evolved.append(gate, evolved.qubits)
            else:
                evolved.compose(gate.definition, evolved.qubits, inplace=True)
            return evolved

        driver_coeff = [x * delta_t for x in self._hopping_rate ] if self._hopping_rate else list(np.ones(len(self.reps)))
        print(driver_coeff)

        for i in range(self.reps):
            gamma = driver_coeff[i]
            for op in self.operators:

                if _is_driver(op):
                    print('--driver--')
                    evolved = _evolve_operator(op, gamma)
                    
                else:
                    print('---cost---')
                    evolved = _evolve_operator(op, delta_t)
                circuit.compose(evolved, circuit.qubits, inplace=True)
        
        return circuit

    