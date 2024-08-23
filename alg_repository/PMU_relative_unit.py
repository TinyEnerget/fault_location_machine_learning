import numpy as np
import pandas as pd

class PMU_relative_units:
        
    def __init__(self,
                 magnitude: pd.DataFrame,
                 measurment_type: str
                 ):
                
        self.magnitude = magnitude
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
        ] / np.sqrt(3) * 10**3

    def measurment(self):

        if self.measurment_type == 'voltage':
            return self.vectors_voltage_calculation(
                self.magnitude
            )

        elif self.measurment_type == 'current':
            return self.vectors_current_calculation(
                self.magnitude
            )

        else:
            raise Exception('Measurement type not recognized')

    def vectors_voltage_calculation(self, 
                             magnitude: pd.DataFrame):
        keys = magnitude.columns
        for key in keys:
            SteadyStateMagnitude = self.SteadystateNormalCalculation(
                magnitude, key
            )
            MeanNominalVoltage = self.MeanNominalVoltageCalculation(
                SteadyStateMagnitude
            )
            magnitude[key] = magnitude[key] / MeanNominalVoltage
        return magnitude
    
    def vectors_current_calculation(self, 
                             magnitude: pd.DataFrame):
        keys = magnitude.columns
        for key in keys:
            SteadyStateMagnitude = self.SteadystateNormalCalculation(
                magnitude, key
            )
            magnitude[key] = magnitude[key] / SteadyStateMagnitude
        return magnitude

    def SteadystateNormalCalculation(self,
                                     magnitude: pd.DataFrame,
                                     key: str):
        
        idx = 0
        tmp_list = []
        idx_list = []
        SteadyStateMagnitude = []
        for item in magnitude.diff()[key]:
            if np.abs(np.round(item, decimals=0)) <= 0 and idx != 0:
                tmp_list.append(idx)
                SteadyStateMagnitude.append(magnitude[key][idx-1])
                idx_list.append(idx-1)

            if tmp_list != []:
                if np.abs(tmp_list[-1]-idx) == 1:
                    print(tmp_list)
                    break
            idx += 1
    
        SteadyStateMagnitude = np.sqrt(np.mean(
            np.square(
                np.array(
                    SteadyStateMagnitude
                    ))))

        return SteadyStateMagnitude
                                     
    def MeanNominalVoltageCalculation(self, 
                                    SteadyStateMagnitude: float) -> float:
        for item in self.MeanNominalPhaseVoltages:
            if SteadyStateMagnitude <= 1.2*item and SteadyStateMagnitude >= 0.8*item:
                MeanNominalVoltage = item
                break
        return MeanNominalVoltage

    