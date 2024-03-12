# fpu/examples/Error_Correction_Example.py

from fpu.src.error_correction.Error_Correction_Algorithms import HammingCode, ReedSolomonEncoderDecoder
import numpy as np

def hamming_code_example():
    print("Hamming Code Example")
    hc = HammingCode(m=3)  # Create a Hamming code with m=3
    original_data = np.array([1, 0, 1, 0])  # Example data
    print("Original Data:", original_data)
    
    encoded_data = hc.encode(original_data)
    print("Encoded Data:", encoded_data)
    
    # Introduce a single bit error in the encoded data
    encoded_data_with_error = np.copy(encoded_data)
    error_position = 2  # Introducing error at position 2
    encoded_data_with_error[error_position] ^= 1
    print("Encoded Data with Error:", encoded_data_with_error)
    
    decoded_data = hc.decode(encoded_data_with_error)
    print("Decoded Data:", decoded_data)

def reed_solomon_example():
    print("\nReed-Solomon Code Example")
    rs = ReedSolomonEncoderDecoder(nsym=10)  # Initialize Reed-Solomon with 10 error correction symbols
    original_data = b"Hello World"
    print("Original Data:", original_data)
    
    encoded_data = rs.encode(original_data)
    print("Encoded Data:", encoded_data)
    
    # Simulate errors in the encoded data
    encoded_data_with_errors = bytearray(encoded_data)
    encoded_data_with_errors[5] ^= 0x01  # Introduce error
    encoded_data_with_errors[15] ^= 0x02  # Introduce another error
    print("Encoded Data with Errors:", encoded_data_with_errors)
    
    corrected_data = rs.decode(encoded_data_with_errors)
    print("Corrected Data:", corrected_data)

if __name__ == "__main__":
    hamming_code_example()
    reed_solomon_example()
