import numpy as np
import pandas as pd

class PMU_symmetrical_components:
    """
    Calculates the symmetrical components (positive, negative, and zero sequence) from the input phase magnitudes and angles.
    
    The `vectors_calculation` method constructs the phase vectors from the input magnitudes and angles, and then uses the symmetrical component transformation matrix to calculate the symmetrical components.
    
    Args:
        magnitude_A (pd.DataFrame): A DataFrame containing the magnitudes of phase A.
        magnitude_B (pd.DataFrame): A DataFrame containing the magnitudes of phase B.
        magnitude_C (pd.DataFrame): A DataFrame containing the magnitudes of phase C.
        angle_A (pd.DataFrame): A DataFrame containing the angles of phase A.
        angle_B (pd.DataFrame): A DataFrame containing the angles of phase B.
        angle_C (pd.DataFrame): A DataFrame containing the angles of phase C.
    
    Returns:
        None
    """
        
    def __init__(self,
                 magnitude_A: pd.DataFrame,
                 magnitude_B: pd.DataFrame,
                 magnitude_C: pd.DataFrame,
                 angle_A: pd.DataFrame,
                 angle_B: pd.DataFrame,
                 angle_C: pd.DataFrame
                 ):
        """
        Initializes the PMU_symmetrical_components class with the necessary input data.
        
        Args:
            magnitude_A (pd.DataFrame): A DataFrame containing the magnitudes of phase A.
            magnitude_B (pd.DataFrame): A DataFrame containing the magnitudes of phase B.
            magnitude_C (pd.DataFrame): A DataFrame containing the magnitudes of phase C.
            angle_A (pd.DataFrame): A DataFrame containing the angles of phase A.
            angle_B (pd.DataFrame): A DataFrame containing the angles of phase B.
            angle_C (pd.DataFrame): A DataFrame containing the angles of phase C.
        
        Attributes:
            sym_factor (np.ndarray): The symmetrical component transformation matrix.
            zero_seq (pd.DataFrame): A DataFrame to store the zero sequence components.
            zero_seq_angle (pd.DataFrame): A DataFrame to store the angles of the zero sequence components.
            negative_seq (pd.DataFrame): A DataFrame to store the negative sequence components.
            negative_seq_angle (pd.DataFrame): A DataFrame to store the angles of the negative sequence components.
            positive_seq (pd.DataFrame): A DataFrame to store the positive sequence components.
            positive_seq_angle (pd.DataFrame): A DataFrame to store the angles of the positive sequence components.
            magnitude_A (pd.DataFrame): The input magnitudes of phase A.
            magnitude_B (pd.DataFrame): The input magnitudes of phase B.
            magnitude_C (pd.DataFrame): The input magnitudes of phase C.
            angle_A (pd.DataFrame): The input angles of phase A.
            angle_B (pd.DataFrame): The input angles of phase B.
            angle_C (pd.DataFrame): The input angles of phase C.
        """
                
        self.sym_factor = self._symmetrical_factor()
        self.zero_seq = pd.DataFrame()
        self.zero_seq_angle = pd.DataFrame()
        self.negative_seq = pd.DataFrame()
        self.negative_seq_angle = pd.DataFrame()
        self.positive_seq = pd.DataFrame()
        self.positive_seq_angle = pd.DataFrame()
        self.magnitude_A = magnitude_A
        self.magnitude_B = magnitude_B
        self.magnitude_C = magnitude_C
        self.angle_A = angle_A
        self.angle_B = angle_B
        self.angle_C = angle_C


    def _symmetrical_factor(self) -> np.ndarray:
        """
        Calculates the symmetrical component transformation matrix.
        
        The symmetrical component transformation matrix is used to transform a three-phase
        system into its symmetrical components (positive, negative, and zero sequence).
        This method returns the 3x3 transformation matrix.
        """
                
        alfa = np.exp(1j * 2 * np.pi/3)
        sym_factor = np.array([
            [1, 1, 1],
            [1, alfa**2, alfa],
            [1, alfa, alfa**2]
        ])
        return sym_factor
    
    def _vector_construction(self, exp, idx) -> np.ndarray:
        """
        Constructs a vector of complex phase values from the input magnitudes and angles.
        
        Args:
            exp (char): The experiment index.
            idx (int): The data index.
        
        Returns:
            np.ndarray: A 3-element numpy array containing the complex phase vectors.
        """
                
        phase_vectors = np.array([
        self.magnitude_A[exp][idx]*np.exp(1j*self.angle_A[exp][idx]),
        self.magnitude_B[exp][idx]*np.exp(1j*self.angle_B[exp][idx]),
        self.magnitude_C[exp][idx]*np.exp(1j*self.angle_C[exp][idx])
        ])
        return phase_vectors

    def vectors_calculation(self) -> dict:
        """
        Calculates the symmetrical components (positive, negative, and zero sequence) from the input phase magnitudes and angles.
        
        This method iterates through the input data, constructs the complex phase vectors, and then applies the symmetrical component transformation matrix to obtain the symmetrical components. The results are stored in the corresponding DataFrame attributes.
        
        Returns:
            dict: A dictionary containing the calculated symmetrical components, with keys for zero sequence, zero sequence angle, negative sequence, negative sequence angle, positive sequence, and positive sequence angle.
        """
                
                
        sym_factor = self.sym_factor
        for exp in self.magnitude_A.columns.values:
            zero_seq_tmp = []
            zero_seq_angle_tmp = []
            negative_seq_tmp = []
            negative_seq_angle_tmp = []
            positive_seq_tmp = []
            positive_seq_angle_tmp = []

            for idx in range(len(self.magnitude_A[exp])):
                phase_vectors = self._vector_construction(exp, idx)
                sym_comp = 1/3 * np.transpose(phase_vectors.transpose().dot(sym_factor))
                zero_seq_tmp.append([np.abs(sym_comp[0])]) 
                zero_seq_angle_tmp.append([np.angle(sym_comp[0])])
                negative_seq_tmp.append([np.abs(sym_comp[1])])
                negative_seq_angle_tmp.append([np.angle(sym_comp[1])])
                positive_seq_tmp.append([np.abs(sym_comp[2])])
                positive_seq_angle_tmp.append([np.angle(sym_comp[2])])

            self.zero_seq[exp] = pd.DataFrame(zero_seq_tmp)
            self.zero_seq_angle[exp] = pd.DataFrame(zero_seq_angle_tmp)
            self.negative_seq[exp] = pd.DataFrame(negative_seq_tmp)
            self.negative_seq_angle[exp] = pd.DataFrame(negative_seq_angle_tmp)
            self.positive_seq[exp] = pd.DataFrame(positive_seq_tmp)
            self.positive_seq_angle[exp] = pd.DataFrame(positive_seq_angle_tmp)
        
        dict_seq = {
            'zero_seq': self.zero_seq,
            'zero_seq_angle': self.zero_seq_angle,
            'negative_seq': self.negative_seq,
            'negative_seq_angle': self.negative_seq_angle,
            'positive_seq': self.positive_seq,
            'positive_seq_angle': self.positive_seq_angle
        }
        return dict_seq