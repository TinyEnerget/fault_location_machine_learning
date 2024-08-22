import numpy as np
import pandas as pd

class PMU_relative_units:
        
    def __init__(self,
                 magnitude_A: pd.DataFrame,
                 magnitude_B: pd.DataFrame,
                 magnitude_C: pd.DataFrame,
                 measurment_type: str
                 ):
                
        self.magnitude_A = magnitude_A
        self.magnitude_B = magnitude_B
        self.magnitude_C = magnitude_C
        self.measurment_type = measurment_type
        self.MeanNominalPhaseVoltages = [
            3.15,
            6.3,
            10.5,
            37,
            115,
            154,
            230,
            340,
            400,
            515,
            750,
            1150
        ] / np.sqrt(3)

    def measurment(self):

        if self.measurment_type == 'voltage':
            return self.vectors_voltage_calculation()

        elif self.measurment_type == 'current':
            return self.vectors_current_calculation()

        else:
            raise Exception('Measurement type not recognized')

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
            self.magnitude_A[exp][idx],
            self.magnitude_B[exp][idx],
            self.magnitude_C[exp][idx]
            ])
        return phase_vectors

    def SteadystateNormalCalculation(self):
        idx = 0
        tmp_list = []
        idx_list = []
        cur_norm = []
        for item in dif_cur:
            if np.abs(np.round(item, decimals=0)) <= 0 and idx != 0:
                tmp_list.append(idx)
                cur_norm.append(tmp_cur[idx-1])
                idx_list.append(idx-1)

            if tmp_list != []:
                if np.abs(tmp_list[-1]-idx) == 1:
                    print(tmp_list)
                    break
            idx += 1
        

cur_norm = np.array(cur_norm)
                                     
    def MeanNominalVoltageCalculation(self, 
                                    SteadyStateNormal: np.ndarray) -> float:
        for item in self.MeanNominalPhaseVoltages:
            if SteadyStateNormal <= 1.2*item and SteadyStateNormal >= 0.8*item:
                MeanNominalVoltage = item
                break
        return MeanNominalVoltage

    