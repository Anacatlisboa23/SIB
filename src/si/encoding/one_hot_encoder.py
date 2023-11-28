from typing import List, Union
import numpy as np

class OneHotEncoder:
    def __init__(self, padder: str, max_length: Union[int, None] = None):
        self.padder = padder
        self.max_length = max_length
        self.alphabet = None
        self.char_to_index = None
        self.index_to_char = None

    def fit(self, data: List[str]) -> None:
        """
        Fit the encoder to the data and learn the alphabet, char_to_index, and index_to_char.

        Args:
        - data: List of sequences (strings) to learn from.
        """
        # Find unique characters in the sequences to form the alphabet
        self.alphabet = list(set("".join(data)))

        # Create dictionaries mapping characters to unique integers and vice versa
        self.char_to_index = {char: index for index, char in enumerate(self.alphabet)}
        self.index_to_char = {index: char for index, char in enumerate(self.alphabet)}

        # Set max_length if not defined
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in data)

    def transform(self, data: List[str]) -> np.ndarray:
        """
        Encode the sequence to one-hot encoding.

        Args:
        - data: List of sequences (strings) to encode.

        Returns:
        - np.ndarray: One-hot encoded matrices.
        """
        # Trim sequences to max_length, pad with the padding character
        padded_data = [seq[:self.max_length].ljust(self.max_length, self.padder) for seq in data]

        # Encode data to one-hot encoded matrices
        encoded_sequences = np.zeros((len(data), self.max_length, len(self.alphabet)), dtype=int)
        for i, seq in enumerate(padded_data):
            for j, char in enumerate(seq):
                # Check if the character is in the alphabet
                if char in self.char_to_index:
                    encoded_sequences[i, j, self.char_to_index[char]] = 1
                else:
                    # Handle characters not present in the learned alphabet
                    print(f"Warning: Character '{char}' not found in the learned alphabet.")

        return encoded_sequences

    def fit_transform(self, data: List[str]) -> np.ndarray:
        """
        Run fit and then transform.

        Args:
        - data: List of sequences (strings) to learn from and encode.

        Returns:
        - np.ndarray: One-hot encoded matrices.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray) -> List[str]:
        """
        Convert one-hot encoded matrices back to sequences using the index_to_char dictionary.

        Args:
        - data: One-hot encoded matrices.

        Returns:
        - List of sequences (strings).
        """
        decoded_sequences = []
        for matrix in data:
            decoded_seq = "".join(self.index_to_char[np.argmax(row)] for row in matrix)
            decoded_sequences.append(decoded_seq.rstrip(self.padder))
        return decoded_sequences

if __name__ == '__main__':
    # Test your OneHotEncoder class
    sequences = ["abc", "defg", "hij"]
    padder = '*'

    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(padder=padder)

    # Fit and transform the sequences
    encoded_data = encoder.fit_transform(sequences)

    # Print the results
    print("Alphabet:", encoder.alphabet)
    print("Char to Index:", encoder.char_to_index)
    print("Index to Char:", encoder.index_to_char)
    print("Encoded Sequences:")
    print(encoded_data)

    # Inverse transform the encoded sequences
    decoded_sequences = encoder.inverse_transform(encoded_data)
    print("Decoded Sequences:")
    print(decoded_sequences)
