import numpy as np
from scipy.stats import expon

class TrafficModel:
    """Class to model different traffic patterns: CBR, Poisson, and Bursty."""
    
    def __init__(self, model_type, **kwargs):
        """
        Initialize the traffic model.
        
        Args:
            model_type (str): 'CBR', 'Poisson', or 'Bursty'.
            kwargs: Model-specific parameters.
        """
        self.model_type = model_type.lower()
        self.params = kwargs
        self.time = 0  # Current simulation time in seconds
        self.packet_log = []  # List to store (time, size) of generated packets
        
        # Validate and initialize model-specific parameters
        if self.model_type == 'cbr':
            self.packet_size = kwargs.get('packet_size', 1000)  # Bytes
            self.packet_rate = kwargs.get('packet_rate', 10)  # Packets/second
            self.inter_packet_time = 1.0 / self.packet_rate  # Seconds
        elif self.model_type == 'poisson':
            self.avg_packet_rate = kwargs.get('avg_packet_rate', 50)  # Packets/second
            self.packet_size_mean = kwargs.get('packet_size_mean', 1000)  # Bytes
            self.packet_size_std = kwargs.get('packet_size_std', 200)  # Bytes
        elif self.model_type == 'bursty':
            self.on_duration = kwargs.get('on_duration', 1.0)  # Seconds
            self.off_duration = kwargs.get('off_duration', 2.0)  # Seconds
            self.packet_rate_on = kwargs.get('packet_rate_on', 20)  # Packets/second
            self.packet_size = kwargs.get('packet_size', 1000)  # Bytes
            self.is_on = True  # Start in ON state
            self.state_time = 0  # Time spent in current state
        else:
            raise ValueError("Invalid model_type. Choose 'CBR', 'Poisson', or 'Bursty'.")

    def generate_packets(self, duration):
        """
        Generate packets for the given duration and store them in self.packet_log.
        
        Args:
            duration (float): Simulation duration in seconds.
        """
        self.packet_log = []
        self.time = 0
        self.state_time = 0
        self.is_on = True
        
        if self.model_type == 'cbr':
            self._generate_cbr(duration)
        elif self.model_type == 'poisson':
            self._generate_poisson(duration)
        elif self.model_type == 'bursty':
            self._generate_bursty(duration)

    def _generate_cbr(self, duration):
        """Generate CBR packets."""
        while self.time < duration:
            self.packet_log.append((self.time, self.packet_size))
            self.time += self.inter_packet_time

    def _generate_poisson(self, duration):
        """Generate Poisson packets."""
        while self.time < duration:
            inter_arrival = expon.rvs(scale=1.0 / self.avg_packet_rate)
            self.time += inter_arrival
            if self.time < duration:
                packet_size = max(100, np.random.normal(self.packet_size_mean, self.packet_size_std))
                self.packet_log.append((self.time, packet_size))

    def _generate_bursty(self, duration):
        """Generate Bursty packets using ON-OFF model."""
        while self.time < duration:
            if self.is_on:
                inter_packet_time = 1.0 / self.packet_rate_on
                while self.state_time < self.on_duration and self.time < duration:
                    self.packet_log.append((self.time, self.packet_size))
                    self.time += inter_packet_time
                    self.state_time += inter_packet_time
                if self.state_time >= self.on_duration:
                    self.is_on = False
                    self.state_time = 0
            else:
                self.time += self.off_duration
                self.state_time = self.off_duration
                if self.time < duration:
                    self.is_on = True
                    self.state_time = 0

    def get_packets_at_time(self, current_time_ms, time_window=0.001):
        """
        Get packets within a small time window around current_time_ms.
        
        Args:
            current_time_ms (int): Current simulation time in milliseconds.
            time_window (float): Time window in seconds (default 0.001 for 1 ms).
        
        Returns:
            list: List of packet sizes (bytes) within the time window.
        """
        current_time_sec = current_time_ms / 1000.0  # Convert ms to seconds
        return [size for (t, size) in self.packet_log 
                if current_time_sec <= t < current_time_sec + time_window]

    def reset(self):
        """Reset the traffic model state."""
        self.time = 0
        self.packet_log = []
        self.state_time = 0
        self.is_on = True