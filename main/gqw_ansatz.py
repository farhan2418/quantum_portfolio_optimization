r'''
this module covers the ansatz required for the guided quantum walk 
as discussed in the paper `Guided Quantum Walk` by sebastian schulz et. al
This ansatz basically takes two hamiltonians named 'driver' and 'cost'
and performs Suzuki trotterization where each hamiltionian is evloved with 
standard Pauli Gates and performs time dependent hamiltonian simulation with
given trotter steps or reps
'''

import numpy as np
import qiskit.quantum_info

from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.library import BlueprintCircuit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli, SparsePauliOp, PauliList
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter, LieTrotter
from qiskit.exceptions import QiskitError

from helper import validate_operators
class GQWAnsatz(BlueprintCircuit):
    def __init__(
        self,
        cost_operator: (qiskit.quantum_info.Pauli
            | SparsePauliOp
            | list[qiskit.quantum_info.Pauli | SparsePauliOp ]
        ),
        hopping_rate: list[float] | float | None = None, 
        reps: int = 1,
        time: float = 1.0,
        driver = None,
        initial_state: QuantumCircuit | None = None,
        name: str = 'GQW',
        flatten: bool | None = None,
        evolution = None
    ) :
        super().__init__(name = name)

        self._cost_operator = cost_operator
        self._reps = reps
        self._time = time
        self._driver = driver
        self._initial_state = initial_state
        self._hopping_rate = hopping_rate
        self._flatten = flatten
        self._evolution = evolution
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
        return [self.cost_operator, self.driver]
    
    @operators.setter
    def operators(self, operators=None) -> None:
        """Set the operators to be evolved.

        operators (Optional[Union[QuantumCircuit, list]]): The operators to evolve.
            If a circuit is passed, we assume it implements an already evolved operator and thus
            the circuit is not evolved again. Can be a single operator (circuit) or a list of
            operators (and circuits).
        """
        operators = validate_operators(operators)
        self._invalidate()
        self._operators = operators
        if self.num_qubits == 0:
            self.qregs = []
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    @property
    def num_qubits(self) -> int:
        return self._cost_operator.num_qubits
    
    # @num_qubits.setter
    # def num_qubits(self, num_qubits: int) -> None:

    #     if self.num_qubits != num_qubits:
    #         # invalidate the circuit
    #         self._invalidate()
    #         # self.num_qubits = num_qubits
    #         self.qregs = [QuantumRegister(num_qubits, name="q")]
    
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self.cregs = [ClassicalRegister(self.num_qubits, name="meas")]
        return valid

    def _build(self) :
        if self._is_built:
            return
        super()._build()
        
        if self.num_qubits == 0:
            return
        
        if not self._flatten:
            circuit = QuantumCircuit(*self.qregs, name = self.name)
            # circuit = QuantumCircuit(*self.qregs, name = self.name)
        else:
            circuit = self
        
        if self.initial_state:
            circuit.compose(self.initial_state.copy(), inplace= True)
        
        hopping_gammas = ParameterVector("Γ", self.reps)
        coeff = ParameterVector("Δ", self.reps)

        reordered = []
        for rep in range(self.reps):
            reordered.append( (hopping_gammas[rep] ,  coeff[rep]  ))



        for i in range(self.reps):
            gamma, c = reordered[i]
            for op in self.operators:
                
                if self._is_driver(op.paulis):
                    evolved = self._evolve_operator(op, gamma, label = 'Driver')
                    
                else:
                    evolved = self._evolve_operator(op, c, label = 'Cost')


                circuit.compose(evolved, circuit.qubits, inplace=True)
        if not self._flatten:
            try:
                block = circuit.to_gate()
            except QiskitError:
                block = circuit.to_instruction()
            self.append(block, self.qubits, copy=False)
        # this assigning is only required when we invoke build method while computing the minimum eigen value 
        # might need to change this as required
        # reordered = []
        # for r in range(self.reps):
            
        #     reordered.append(self._hopping_rate[r]*delta_t)
        #     reordered.append(delta_t)
        # circuit.assign_parameters(reordered, inplace = False)

    def _is_driver(self, paulis: PauliList | list) -> bool:
        pauli = paulis[0]
        if isinstance(pauli, Pauli):
            return np.any(pauli.x)
        
        return False
    
    def _evolve_operator(self, operator: SparsePauliOp, time, label: str| None = None):
        # evolution = LieTrotter() if self._evolution is None else self._evolution
        evolution = SuzukiTrotter() if self._evolution is None else self._evolution
        gate = PauliEvolutionGate(operator, time, synthesis=evolution, label=label)
        evolved = QuantumCircuit(operator.num_qubits)
        if not self.flatten:
            evolved.append(gate, evolved.qubits)
        else:
            evolved.compose(gate.definition, evolved.qubits, inplace=True)
        return evolved