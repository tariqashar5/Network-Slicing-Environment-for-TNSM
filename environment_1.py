from __future__ import print_function
import numpy as np
import pandas as pd
import random
import math
from gym import spaces
from sps_mat import sps  # Import SPS mechanism for PDR calculation

# Utility Functions
def normalize_custom(data, from_min, from_max, to_min, to_max):
    """Normalize data to a specific range."""
    return (((data - from_min) / (from_max - from_min)) * (to_max - to_min)) + to_min

class VehicularEnvironment:
    def __init__(self, total_vehicles=250):
        # System parameters
        self.total_vehicles = total_vehicles
        self.total_bandwidth = 10e6  # 10 MHz
        self.total_subchannels = 50
        self.slice_types = ['Traffic Safety', 'Autonomous Driving']
        self.packet_sizes = [300, 190]  # in bytes
        self.packet_intervals = [100, 50]  # in ms

        # Environment parameters
        self.vehicle_requests = [0] * self.total_vehicles

        # Resource allocation parameters
        self.subchannels_per_slice = [25, 25]  # Initial allocation (static split)
        self.min_required_rb = [2, 1]  # Min RBs per packet for each slice

        # SPS parameters
        self.reselection_counter = [20, 10]  # For each slice
        self.periodicity = [50, 100]  # For each slice
        self.total_subframes = 1000  # Total subframes for SPS simulation

        # Mobility trace
        self.trace = None
        self.positions = np.zeros(self.total_vehicles)  # Placeholder for positions
        self.trace_index = np.zeros(self.total_vehicles, dtype=int)  # Index in trace for each vehicle

        self.generate_synthetic_trace()

        # State and action spaces
        low_obs = np.array([0, 0, 0, 0])  # Min: vehicles and subchannels per slice
        high_obs = np.array([self.total_vehicles, self.total_vehicles, self.total_subchannels, self.total_subchannels])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initial state
        self.state = None
        self.reset()

    def load_trace(self, trace_file):
        """Load mobility trace from a file."""
        self.trace = pd.read_csv(trace_file)
        if 'vehicle_id' not in self.trace.columns or 'time' not in self.trace.columns or 'position' not in self.trace.columns:
            raise ValueError("Trace file must contain 'vehicle_id', 'time', and 'position' columns.")

    def generate_synthetic_trace(self):
        """Generate synthetic mobility trace with traffic flow models."""
        simulation_time = 3600  # 1 hour
        time_step = 1  # 1 second
        data = []

        for vehicle_id in range(self.total_vehicles):
            # Define traffic type (urban/highway) and assign speed using the NaSch model
            traffic_type = np.random.choice(['urban', 'highway'])
            if traffic_type == 'urban':
                speed = np.random.normal(11.1, 2.8)  # Mean 11.1 m/s, SD 2.8 (urban traffic)
            else:
                speed = np.random.normal(27.7, 3.3)  # Mean 27.7 m/s, SD 3.3 (highway traffic)

            start_time = np.random.uniform(0, simulation_time)  # Vehicle starts at a random time

            for t in range(int(start_time), simulation_time, time_step):
                position = speed * (t - start_time)  # Calculate position based on speed and time
                data.append([vehicle_id, t, position, speed, traffic_type])

        # Save trace with vehicle_id, time, position, speed, and traffic type
        self.trace = pd.DataFrame(data, columns=['vehicle_id', 'time', 'position', 'speed', 'traffic_type'])

    def save_trace(self, filename='synthetic_trace.csv'):
        """Save the generated synthetic trace to a CSV file."""
        if self.trace is not None:
            self.trace.to_csv(filename, index=False)
        else:
            print("No trace available to save!")

    def update_positions(self):
        """Update vehicle positions based on the trace data."""
        for i in range(self.total_vehicles):
            vehicle_trace = self.trace[self.trace['vehicle_id'] == i]
            if self.trace_index[i] >= len(vehicle_trace):
                self.trace_index[i] = 0  # Loop back to the start of the trace

            self.positions[i] = vehicle_trace.iloc[self.trace_index[i]]['position']
            self.trace_index[i] += 1

    def generate_requests(self):
        """Generate network requests dynamically based on traffic density."""
        density_factor = len([pos for pos in self.positions if pos < 500]) / self.total_vehicles  # Simplified density
        for i in range(self.total_vehicles):
            if density_factor > 0.7:  # Higher density leads to more requests
                self.vehicle_requests[i] = np.random.choice([0, 1], p=[0.6, 0.4])  # Slice distribution
            else:
                self.vehicle_requests[i] = np.random.choice([0, 1], p=[0.4, 0.6])  # Lower density

    def step(self, action):
        """Execute one time step in the environment."""
        # Normalize and apply action
        actions = [normalize_custom(a, -1, 1, 0, self.total_subchannels // 2) for a in action]
        self.subchannels_per_slice[0] = math.floor(actions[0])
        self.subchannels_per_slice[1] = math.ceil(self.total_subchannels - self.subchannels_per_slice[0])

        # Constrain resource allocation
        self.subchannels_per_slice[0] = max(self.min_required_rb[0], min(self.subchannels_per_slice[0], self.total_subchannels))
        self.subchannels_per_slice[1] = max(self.min_required_rb[1], min(self.subchannels_per_slice[1], self.total_subchannels))

        # Update vehicle positions using trace
        self.update_positions()

        # Generate requests dynamically
        self.generate_requests()

        # Calculate PDR and SE using SPS
        collision_probability_slice1 = sps(
            sum([1 for req in self.vehicle_requests if req == 0]),
            self.subchannels_per_slice[0],
            self.total_subframes,
            self.reselection_counter[0],
            self.periodicity[0]
        )

        collision_probability_slice2 = sps(
            sum([1 for req in self.vehicle_requests if req == 1]),
            self.subchannels_per_slice[1],
            self.total_subframes,
            self.reselection_counter[1],
            self.periodicity[1]
        )

        pdr_slice1 = 1 - collision_probability_slice1
        pdr_slice2 = 1 - collision_probability_slice2

        throughput_slice1 = pdr_slice1 * sum([1 for req in self.vehicle_requests if req == 0]) * self.packet_sizes[0] * 8  # bits
        throughput_slice2 = pdr_slice2 * sum([1 for req in self.vehicle_requests if req == 1]) * self.packet_sizes[1] * 8  # bits

        se_slice1 = throughput_slice1 / (self.subchannels_per_slice[0] * 180e3) if self.subchannels_per_slice[0] > 0 else 0
        se_slice2 = throughput_slice2 / (self.subchannels_per_slice[1] * 180e3) if self.subchannels_per_slice[1] > 0 else 0

        # Calculate reward
        reward = self.calculate_reward([pdr_slice1, pdr_slice2], [se_slice1, se_slice2])

        # Update state
        self.state = [
            sum([1 for req in self.vehicle_requests if req == 0]),  # Vehicles for slice 1
            sum([1 for req in self.vehicle_requests if req == 1]),  # Vehicles for slice 2
            self.subchannels_per_slice[0],
            self.subchannels_per_slice[1]
        ]

        # Define done condition (optional)
        done = False

        # Info dictionary with metrics
        info = {
            'pdr_slice1': pdr_slice1,
            'pdr_slice2': pdr_slice2,
            'se_slice1': se_slice1,
            'se_slice2': se_slice2
        }

        return np.array(self.state, dtype=np.float32), reward, done, info

    def reset(self):
        """Reset the environment to its initial state."""
        self.positions = np.zeros(self.total_vehicles)
        self.vehicle_requests = [0] * self.total_vehicles
        self.subchannels_per_slice = [25, 25]  # Reset resource allocation
        self.trace_index = np.zeros(self.total_vehicles, dtype=int)  # Reset trace index

        self.state = [
            self.total_vehicles // 2,  # Equal initial split
            self.total_vehicles // 2,
            self.subchannels_per_slice[0],
            self.subchannels_per_slice[1]
        ]
        return np.array(self.state, dtype=np.float32)

    def render(self):
        """Render the current state of the environment."""
        print(f"State: {self.state}")

    def calculate_kpis(self):
        """Calculate Packet Delivery Ratio (PDR) and Spectral Efficiency (SE) using SPS."""
        collision_probability = []

        for i in range(2):
            collision_probability.append(
                sps(
                    sum([1 for req in self.vehicle_requests if req == i]),
                    self.subchannels_per_slice[i],
                    self.total_subframes,
                    self.reselection_counter[i],
                    self.periodicity[i]
                )
            )

        pdr = [1 - p for p in collision_probability]

        # SE: Throughput / Bandwidth
        throughput = [
            pdr[i] * sum([1 for req in self.vehicle_requests if req == i]) * self.packet_sizes[i] * 8 for i in range(2)  # bits
        ]
        se = [
            throughput[i] / (self.subchannels_per_slice[i] * 180e3) if self.subchannels_per_slice[i] > 0 else 0 for i in range(2)
        ]

        return pdr, se

    def calculate_reward(self, pdr, se):
        """Calculate reward based on KPIs."""
        # Weighted sum of PDR and SE
        weights = [0.7, 0.3]  # Priority to PDR
        reward = sum(weights[i] * (pdr[i] + se[i]) for i in range(2))
        return reward
