# fpu/examples/Logical_Qubits_Example.py

from qiskit import Aer, execute
from fpu.src.quantum.Logical_Qubit_Management import LogicalQubitManager
import numpy as np

def run_logical_qubit_example():
    num_qubits = 3  # Number of qubits for the logical qubit
    lqm = LogicalQubitManager(num_qubits=num_qubits)

    # Initialize a simple quantum state to encode
    initial_state = [1/np.sqrt(2), 1/np.sqrt(2)]  # Superposition state (|0> + |1>)/sqrt(2)

    print("Encoding a logical qubit...")
    lqm.encode_logical_qubit(initial_state)

    print("Applying a logical X gate to the logical qubit...")
    lqm.apply_logical_gate('X')

    print("Decoding the logical qubit (with error correction)...")
    corrected_state = lqm.decode_logical_qubit()

    # Simulate and get the state vector of the corrected logical qubit
    backend = Aer.get_backend('statevector_simulator')
    job = execute(lqm.qc, backend)
    result = job.result()
    output_state = result.get_statevector()

    print("Final state vector of the logical qubit:", output_state)

if __name__ == "__main__":
    run_logical_qubit_example()
