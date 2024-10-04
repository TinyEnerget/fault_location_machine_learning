import numpy as np
import pandas as pd
from numba import njit
from concurrent.futures import ThreadPoolExecutor

class PMU_relative_units:
    def __init__(self, magnitude: pd.DataFrame, measurement_type: str):
        self.magnitude = magnitude
        self.measurement_type = measurement_type
        self.MeanNominalPhaseVoltages = np.array(
            [3.15, 6.3, 10.5, 37, 115, 154, 230, 340, 400, 515, 750, 1150]
            ) / np.sqrt(3) * 10**3

    def measurement(self):
        if self.measurement_type == 'voltage':
            return self.vectors_voltage_calculation(self.magnitude)
        elif self.measurement_type == 'current':
            return self.vectors_current_calculation(self.magnitude)
        else:
            raise Exception('Measurement type not recognized')

    def vectors_voltage_calculation(self, magnitude: pd.DataFrame):
        steady_state_magnitudes = self.parallel_steady_state_calculation(magnitude)
        return magnitude.div(steady_state_magnitudes)

    def vectors_current_calculation(self, magnitude: pd.DataFrame):
        steady_state_magnitudes = self.parallel_steady_state_calculation(magnitude)
        return magnitude.div(steady_state_magnitudes)

    def parallel_steady_state_calculation(self, magnitude: pd.DataFrame):
        with ThreadPoolExecutor() as executor:
            steady_state_magnitudes = list(executor.map(
                self.SteadystateNormalCalculation,
                [magnitude[col].values for col in magnitude.columns]
            ))
        return pd.Series(steady_state_magnitudes, index=magnitude.columns)

    @staticmethod
    def SteadystateNormalCalculation(column):
        return _steady_state_calculation(column)

    def MeanNominalVoltageCalculation(self, SteadyStateMagnitude: float) -> float:
        mask = (SteadyStateMagnitude <= 1.2 * self.MeanNominalPhaseVoltages) & (SteadyStateMagnitude >= 0.8 * self.MeanNominalPhaseVoltages)
        if np.any(mask):
            return self.MeanNominalPhaseVoltages[mask][0]
        return np.nan

@njit
def _steady_state_calculation(column):
    diff = np.diff(column)
    steady_state_mask = np.abs(np.round(diff, decimals=0)) <= 0
    steady_state_indices = np.where(steady_state_mask)[0]
    if len(steady_state_indices) > 0:
        steady_state_values = column[steady_state_indices]
        return np.sqrt(np.mean(np.square(steady_state_values)))
    return np.nan