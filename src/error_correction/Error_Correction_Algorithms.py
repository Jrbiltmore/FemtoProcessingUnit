# fpu/src/error_correction/Error_Correction_Algorithms.py

import numpy as np
import reedsolo
from scipy.linalg import hadamard
from sklearn.metrics import hamming_loss

class HammingCode:
    def __init__(self, m=3):
        self.m = m
        self.n = 2 ** m - 1
        self.k = self.n - m
        self.generator_matrix = self.generate_generator_matrix()
        self.parity_check_matrix = self.generate_parity_check_matrix()
        self.syndrome_table = self.generate_syndrome_table()

    def generate_generator_matrix(self):
        # Generate identity matrix of size k
        I_k = np.eye(self.k, dtype=int)
        # Generate parity matrix P
        P = [list((bin(i+1)[2:]).zfill(self.m)) for i in range(self.k)]
        P = np.array(P, dtype=int).T
        # Concatenate I_k and P to get the generator matrix G
        G = np.concatenate((I_k, P), axis=1)
        return G

    def generate_parity_check_matrix(self):
        # Generate identity matrix of size m
        I_m = np.eye(self.m, dtype=int)
        # The parity check matrix H is [P.T | I_m]
        H = np.concatenate((self.generator_matrix[:, self.k:].T, I_m), axis=1)
        return H

    def generate_syndrome_table(self):
        syndromes = {}
        for i in range(1, self.n + 1):
            error_vector = np.zeros(self.n, dtype=int)
            error_vector[i-1] = 1
            syndrome = np.dot(error_vector, self.parity_check_matrix.T) % 2
            syndrome_key = tuple(syndrome)
            syndromes[syndrome_key] = i
        return syndromes

    def encode(self, data):
        if len(data) != self.k:
            raise ValueError(f"Data length must be {self.k}")
        encoded_data = np.dot(data, self.generator_matrix) % 2
        return encoded_data

    def decode(self, received):
        if len(received) != self.n:
            raise ValueError(f"Received data length must be {self.n}")
        syndrome = np.dot(received, self.parity_check_matrix.T) % 2
        syndrome_key = tuple(syndrome)
        error_position = self.syndrome_table.get(syndrome_key, 0)
        if error_position:
            received[error_position-1] ^= 1  # Correct the error
        return received[:self.k]  # Return the original data

class ReedSolomonEncoderDecoder:
    """
    A simple wrapper around the Reed-Solo library for Reed-Solomon encoding and decoding.
    This class is designed for handling byte data, making it suitable for practical applications.
    """
    def __init__(self, nsym):
        """
        Initialize the Reed-Solomon encoder/decoder.
        :param nsym: Number of Reed-Solomon symbols (error correction bytes).
        """
        self.nsym = nsym
        self.rs = reedsolo.RSCodec(self.nsym)

    def encode(self, data):
        """
        Encode data using Reed-Solomon error correction.
        :param data: Input data as bytes.
        :return: Encoded data with error correction codes.
        """
        encoded_data = self.rs.encode(data)
        return encoded_data

    def decode(self, data):
        """
        Decode and correct errors in the given data using Reed-Solomon.
        :param data: Data with potential errors as bytes.
        :return: Corrected data, stripped of error correction codes.
        """
        try:
            corrected_data, _ = self.rs.decode(data)
            return corrected_data
        except reedsolo.ReedSolomonError as e:
            print(f"Error correcting data: {e}")
            return None

# Example usage of Reed-Solomon Encoder/Decoder
if __name__ == "__main__":
    rs_encoder_decoder = ReedSolomonEncoderDecoder(nsym=10)  # Initialize with 10 error correction bytes

    # Example data
    data = b"Hello, world!"
    print("Original data:", data)

    # Encoding data
    encoded_data = rs_encoder_decoder.encode(data)
    print("Encoded Data:", encoded_data)

    # Simulate errors in data
    encoded_data_with_errors = bytearray(encoded_data)
    encoded_data_with_errors[5] ^= 0x01  # Introducing a simple error
    encoded_data_with_errors[10] ^= 0x02  # Introducing another error

    # Decoding and correcting errors
    corrected_data = rs_encoder_decoder.decode(bytes(encoded_data_with_errors))
    if corrected_data:
        print("Corrected Data:", corrected_data)
    else:
        print("Error correcting the data was unsuccessful.")

    # Hamming Code Example
    hc = HammingCode(m=3)
    original_data = np.array([1, 0, 1, 0])
    encoded_data = hc.encode(original_data)
    print("Encoded Data:", encoded_data)

    # Introduce a single bit error
    encoded_data[2] ^= 1
    print("Received Data (with error):", encoded_data)

    decoded_data = hc.decode(encoded_data)
    print("Decoded Data:", decoded_data)

    # Calculate Hamming Loss
    loss = hamming_loss(original_data, decoded_data)
    print("Hamming Loss:", loss)
