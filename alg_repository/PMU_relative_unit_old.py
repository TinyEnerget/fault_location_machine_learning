import numpy as np
import pandas as pd

class PMU_relative_units:
    """
    Provides a class `PMU_relative_units` that calculates voltage and current measurements relative to nominal values.
    
    The class takes a `pd.DataFrame` of magnitude measurements and a measurement type ('voltage' or 'current') as input. It then calculates the steady-state magnitude and mean nominal voltage/current for the measurements, and returns the measurements normalized by the mean nominal values.
    
    The class has the following methods:
    - `__init__(self, magnitude: pd.DataFrame, measurment_type: str)`: Initializes the class with the input magnitude measurements and measurement type.
    - `measurment(self)`: Calculates the normalized voltage or current measurements based on the measurement type.
    - `vectors_voltage_calculation(self, magnitude: pd.DataFrame)`: Calculates the normalized voltage measurements.
    - `vectors_current_calculation(self, magnitude: pd.DataFrame)`: Calculates the normalized current measurements.
    - `SteadystateNormalCalculation(self, magnitude: pd.DataFrame, key: str)`: Calculates the steady-state magnitude for a given measurement.
    - `MeanNominalVoltageCalculation(self, SteadyStateMagnitude: float)`: Calculates the mean nominal voltage for a given steady-state magnitude.
    """
                
    def __init__(self,
                 magnitude: pd.DataFrame,
                 measurment_type: str
                 ):
        """
        Initializes the PMU_relative_units class with the input magnitude measurements and measurement type.
        
        Args:
            magnitude (pd.DataFrame): A DataFrame containing the magnitude measurements.
            measurment_type (str): The type of measurement, either 'voltage' or 'current'.
        
        Attributes:
            magnitude (pd.DataFrame): The input magnitude measurements.
            measurment_type (str): The type of measurement.
            MeanNominalPhaseVoltages (list): A list of mean nominal phase voltages.
        """
                                
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
        """
        Calculates the normalized voltage or current measurements based on the measurement type.
        
        Returns:
            pd.DataFrame: The normalized voltage or current measurements.
        
        Raises:
            Exception: If the measurement type is not recognized.
        """
                
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
        """
        Calculates the normalized voltage measurements based on the steady-state magnitude and mean nominal voltage.
        
        Args:
            magnitude (pd.DataFrame): The input voltage magnitude measurements.
        
        Returns:
            pd.DataFrame: The normalized voltage measurements.
        """
                
        keys = magnitude.columns
        for key in keys:
            SteadyStateMagnitude = self.SteadystateNormalCalculation(
                magnitude, key
            )
            #MeanNominalVoltage = self.MeanNominalVoltageCalculation(
            #    SteadyStateMagnitude
            #)
            magnitude[key] = magnitude[key] / SteadyStateMagnitude
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
        """
        Calculates the steady-state magnitude of a measurement.
        
        Args:
            magnitude (pd.DataFrame): The input measurement magnitude.
            key (str): The column key of the measurement.
        
        Returns:
            float: The steady-state magnitude of the measurement.
        """
                
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
                    #print(tmp_list)
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
        """
        Calculates the mean nominal voltage based on the steady-state magnitude.
        
        Args:
            SteadyStateMagnitude (float): The steady-state magnitude of the measurement.
        
        Returns:
            float: The mean nominal voltage.
        """
                
        for item in self.MeanNominalPhaseVoltages:
            if SteadyStateMagnitude <= 1.2*item and SteadyStateMagnitude >= 0.8*item:
                MeanNominalVoltage = item
                break
        return MeanNominalVoltage

    