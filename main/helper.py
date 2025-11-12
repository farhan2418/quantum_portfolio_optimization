"""
this file contains the code for few helper functions such as 
- forming the QUBO 
- conversion to Ising Hamiltonian
and these are the essentitals to solve such problems in quantum computer
"""
import numpy as np

from scipy.optimize import brentq

def calc_gamma(t, T, lambda_vec: list | np.ndarray):
    l1, l2, l3, l4, l5, l6 = lambda_vec
    s = t/T

    def _x_tau_minus_s(tau):
        x_tau = 3* l1 *((1- tau)**2) * tau \
              + 3 *l3*(1- tau) * (tau**2) \
                  + tau**3 
        return (x_tau - s)
    
    if s <= 0.0:
        tau = 0.0
    elif s >= 1.0:
        tau = 1.0
    else:
        try:
            tau = brentq(_x_tau_minus_s, 0.0, 1.0)
        except ValueError:
            tau = s
    
    y_tau = ((1 - tau)**3) + \
        (3 * l2 *   ((1- tau)**2) * tau) + \
            (3 * l4 * (1 - tau) * (tau**2)) 

    start_gamma = 10 ** (2.0 * l5)
    end_gamma = 10 ** (-3.0 * l6)
    # print(start_gamma, end_gamma)
    gamma = y_tau * start_gamma + (1 - y_tau) * end_gamma

    return gamma

def compare_measurements(candidate, current_best):
    if candidate["value"] < current_best["value"]:
        return True
    elif candidate["value"] == current_best["value"]:
        return candidate["probability"] > current_best["probability"]
    return False

def validate_operators(operators):
    if not isinstance(operators, list):
        operators = [operators]

    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
            raise ValueError("All operators must act on the same number of qubits.")

    return operators