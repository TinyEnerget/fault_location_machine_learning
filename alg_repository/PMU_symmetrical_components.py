import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor


class PMU_symmetrical_components:
 
    def __init__(self,
                 magnitude_A: pd.DataFrame,
                 magnitude_B: pd.DataFrame,
                 magnitude_C: pd.DataFrame,
                 angle_A: pd.DataFrame,
                 angle_B: pd.DataFrame,
                 angle_C: pd.DataFrame
                 ):
                
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

    @staticmethod
    def _symmetrical_factor() -> np.ndarray:
        alfa = np.exp(1j * 2 * np.pi/3)
        sym_factor = np.array([
            [1, 1, 1],
            [1, alfa**2, alfa],
            [1, alfa, alfa**2]
        ])
        return sym_factor
          #@staticmethod
          #@njit
          #def _vector_construction(
          #    magnitude_A: np.ndarray,
          #    magnitude_B: np.ndarray,
          #    magnitude_C: np.ndarray,
          #    angle_A: np.ndarray,
          #    angle_B: np.ndarray,
          #    angle_C: np.ndarray) -> np.ndarray:
        #
          #    phase_vectors = np.empty((3, len(magnitude_A)), dtype=np.complex128)
          #    phase_vectors[0] = magnitude_A * np.exp(1j * angle_A)
          #    phase_vectors[1] = magnitude_B * np.exp(1j * angle_B)
          #    phase_vectors[2] = magnitude_C * np.exp(1j * angle_C)
          #    return phase_vectors
    @staticmethod
    @njit
    def _calculate_symmetrical_components(phase_vectors, sym_factor):
        return 1/3 * phase_vectors.transpose().dot(sym_factor)

    def _process_experiment(self, exp):
        phase_vectors = _vector_construction(
            self.magnitude_A[exp].values,
            self.magnitude_B[exp].values,
            self.magnitude_C[exp].values,
            self.angle_A[exp].values,
            self.angle_B[exp].values,
            self.angle_C[exp].values
        )
        sym_comp = self._calculate_symmetrical_components(phase_vectors, self.sym_factor)
        
        return {
            'zero_seq': np.abs(sym_comp[:, 0]),
            'zero_seq_angle': np.angle(sym_comp[:, 0]),
            'negative_seq': np.abs(sym_comp[:, 1]),
            'negative_seq_angle': np.angle(sym_comp[:, 1]),
            'positive_seq': np.abs(sym_comp[:, 2]),
            'positive_seq_angle': np.angle(sym_comp[:, 2])
        }

    def vectors_calculation(self) -> dict:
        results = [self._process_experiment(exp) for exp in self.magnitude_A.columns]
        
        dict_seq = {
            'zero_seq': pd.DataFrame({exp: result['zero_seq'] for exp, result in zip(self.magnitude_A.columns, results)}),
            'zero_seq_angle': pd.DataFrame({exp: result['zero_seq_angle'] for exp, result in zip(self.magnitude_A.columns, results)}),
            'negative_seq': pd.DataFrame({exp: result['negative_seq'] for exp, result in zip(self.magnitude_A.columns, results)}),
            'negative_seq_angle': pd.DataFrame({exp: result['negative_seq_angle'] for exp, result in zip(self.magnitude_A.columns, results)}),
            'positive_seq': pd.DataFrame({exp: result['positive_seq'] for exp, result in zip(self.magnitude_A.columns, results)}),
            'positive_seq_angle': pd.DataFrame({exp: result['positive_seq_angle'] for exp, result in zip(self.magnitude_A.columns, results)})
        }
        return dict_seq
    
@staticmethod
@njit
def _vector_construction(
    magnitude_A: np.ndarray,
    magnitude_B: np.ndarray,
    magnitude_C: np.ndarray,
    angle_A: np.ndarray,
    angle_B: np.ndarray,
    angle_C: np.ndarray) -> np.ndarray:

    phase_vectors = np.empty((3, len(magnitude_A)), dtype=np.complex128)
    phase_vectors[0] = magnitude_A * np.exp(1j * angle_A)
    phase_vectors[1] = magnitude_B * np.exp(1j * angle_B)
    phase_vectors[2] = magnitude_C * np.exp(1j * angle_C)
    return phase_vectors