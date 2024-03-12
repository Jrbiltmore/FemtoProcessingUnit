# fpu/examples/Quantum_Computing_Example.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

def run_quantum_teleportation():
    # Create a quantum circuit with 3 qubits and 3 classical bits
    qc = QuantumCircuit(3, 3)

    # Step 1: Create entanglement between qubit 1 and qubit 2
    qc.h(1)
    qc.cx(1, 2)

    # Step 2: Prepare the initial state of qubit 0 that we want to teleport
    qc.rx(1.23, 0)  # Applying a rotation for example state preparation

    # Step 3: Perform the teleportation protocol
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])  # Measure qubits 0 and 1
    qc.cx(1, 2)
    qc.cz(0, 2)

    # Step 4: Measure the final state of qubit 2
    qc.measure(2, 2)

    # Execute the quantum circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts(qc)

    # Plot the results
    plot_histogram(counts)
    plt.title("Quantum Teleportation")
    plt.show()

if __name__ == "__main__":
    run_quantum_teleportation()
