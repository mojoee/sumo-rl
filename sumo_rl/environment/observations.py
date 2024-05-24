"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )
class CustomObservationFunction(ObservationFunction):
    """Custom observation function for traffic signals with neighbor density classification."""

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation with density classification."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()

        # Calculate the average density of neighboring traffic signals
        total_density = []
        for neighbor_ts_id in self.ts.neighbors:
            neighbor_ts = self.ts.env.traffic_signals[neighbor_ts_id]
            neighbor_density = neighbor_ts.get_lanes_density()
            total_density.extend(neighbor_density)
        average_neighbors_density = np.mean(total_density) if total_density else 0.0

        # Classify the average density into three levels
        if average_neighbors_density <= 0.33:
            density_classification = 0  # Low density
        elif average_neighbors_density <= 0.66:
            density_classification = 0.5  # Medium density
        else:
            density_classification = 1  # High density

        # Combine current traffic signal's data with the density classification
        observation = np.array(phase_id + min_green + density + queue + [density_classification], dtype=np.float32)
        return observation
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space adjusted for density classification."""
        # Adjust the size of observation space to accommodate the density classification
        num_phases = self.ts.num_green_phases  # Number of phases for the current traffic signal
        num_lanes = 2 * len(self.ts.lanes)  # Current traffic signal's lanes
        return spaces.Box(
            low=np.zeros(num_phases + 1 + num_lanes + 1, dtype=np.float32),  # +1 for the density classification
            high=np.ones(num_phases + 1 + num_lanes + 1, dtype=np.float32),
        )
