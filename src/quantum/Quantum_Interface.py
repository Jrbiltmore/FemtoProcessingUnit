# fpu/src/quantum/Quantum_Interface.py

from qiskit import IBMQ, Aer, execute, QuantumCircuit
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

class QuantumInterface:
    """
    Interface to execute quantum circuits using both local simulator and real quantum devices.
    """
    def __init__(self, use_ibm_q=False, api_token=None):
        self.use_ibm_q = use_ibm_q
        if use_ibm_q and api_token:
            IBMQ.save_account(api_token, overwrite=True)
            IBMQ.load_account()
            self.provider = IBMQ.get_provider(hub='ibm-q')
        self.backend = Aer.get_backend('qasm_simulator')

    def set_backend(self, backend_name):
        """
        Set the backend for executing the quantum circuit.
        :param backend_name: Name of the backend ('qasm_simulator' for simulation, or real device names for execution on IBM Q)
        """
        if self.use_ibm_q:
            self.backend = self.provider.get_backend(backend_name)
        else:
            self.backend = Aer.get_backend(backend_name)

    def execute_circuit(self, qc, shots=1024):
        """
        Executes a quantum circuit.
        :param qc: The quantum circuit to execute.
        :param shots: Number of repetitions of each circuit, for sampling.
        :return: The result of the execution.
        """
        job = execute(qc, self.backend, shots=shots)
        if self.use_ibm_q:
            job_monitor(job)
        result = job.result()
        counts = result.get_counts(qc)
        return counts

    def plot_results(self, counts):
        """
        Plots the results of the quantum circuit execution.
        :param counts: The result counts from the circuit execution.
        """
        plot_histogram(counts)
        plt.show()

# Example usage
if __name__ == "__main__":
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # Use local simulator
    interface = QuantumInterface(use_ibm_q=False)
    result = interface.execute_circuit(qc)
    interface.plot_results(result)

    # To use IBM Q, uncomment the following lines and provide your IBM Q Experience API Token
    # api_token = "YOUR_IBM_Q_API_TOKEN_HERE"
    # interface = QuantumInterface(use_ibm_q=True, api_token=api_token)
    # interface.set_backend('ibmq_quito')  # Example IBM Q backend
    # result_ibm_q = interface.execute_circuit(qc)
    # interface.plot_results(result_ibm_q)
