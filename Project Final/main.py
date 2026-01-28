"""
SpiNNaker 2 HAR Project - Parameter Sweep Implementation
Adapts processed Kaggle data to simulate a time-series input for SNN.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spinnaker2 import hardware, snn
from spinnaker2.experiment_backends.base_experiment_backend import ExperimentBackendType

# --- CONFIGURATION ---
# Use the IP address provided by your lab setup (e.g., 192.168.1.53 or 192.168.4.17)
# Check the uploaded file 'lif_transfer_curve_multi_run_fast.py' for IP examples
S2_IP = os.environ.get("S2_IP", "192.168.1.53") 

# --- 1. DATA PREPARATION (The Workaround) ---
def load_pseudo_timeseries(filename='train.csv', max_samples=1000):
    """
    Loads processed data and chains the 'mean acceleration' feature
    to create a pseudo-continuous signal for Delta Encoding.
    """
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)
    
    # In the Kaggle dataset, the first column is usually tBodyAcc-mean()-X
    # We use this as a proxy for the raw X-axis signal.
    # We also check if 'Activity' column exists for labels.
    
    if 'Activity' in df.columns:
        # Extract signal proxy (Column 0)
        signal_proxy = df.iloc[:, 0].values
        labels = df['Activity'].values
    else:
        # Fallback if column names are missing
        signal_proxy = df.iloc[:, 0].values
        labels = df.iloc[:, -1].values

    # Normalize signal to range [-1, 1] for better Delta Encoding
    signal_proxy = (signal_proxy - np.mean(signal_proxy)) / np.std(signal_proxy)
    
    # Limit data size for faster simulation testing
    return signal_proxy[:max_samples], labels[:max_samples]

# --- 2. DELTA ENCODING ---
def delta_encoding(signal, threshold):
    """
    Converts analog signal to spike times.
    Logic: Fire spike if |current - previous| > threshold
    Returns: A dictionary format required by 'spike_list' model {neuron_id: [times]}
    """
    spike_times = []
    prev_val = signal[0]
    dt = 10.0 # Virtual time step (ms)
    
    for i, val in enumerate(signal):
        if np.abs(val - prev_val) > threshold:
            spike_times.append(int(i * dt)) # Convert to integer time step
            prev_val = val
            
    # Return as dictionary for neuron 0 (since we have 1 input sensor channel)
    return {0: spike_times}

# --- 3. MAIN EXPERIMENT ---
def run_simulation():
    # Load Data
    signal, labels = load_pseudo_timeseries()
    print(f"Data Loaded: {len(signal)} samples.")

    # Define thresholds to sweep (The core of your 'Parameter Sweep Study')
    thresholds = [0.1, 0.5, 1.0, 1.5]
    results_spike_count = []
    
    # --- NETWORK SETUP (Static Architecture) ---
    # We define the network structure once, similar to 'lif_transfer_curve_multi_run.py'
    
    # 1. Input Layer (Dummy params initially, updated in loop)
    # Uses 'spike_list' model as seen in 'lif_curr_exp.py'
    stim = snn.Population(size=1, neuron_model="spike_list", params={0: []}, name="Input_Sensor")
    
    # 2. Processing Layer (LIF Neurons)
    # Parameters adapted from 'lif_curr_exp.py'
    neuron_params = {
        "threshold": 10.0,
        "alpha_decay": 0.9,
        "i_offset": 0.0,
        "v_reset": 0.0,
        "reset": "reset_by_subtraction"
    }
    # Using 'lif_curr_exp' model
    pop_process = snn.Population(size=5, neuron_model="lif_curr_exp", params=neuron_params, name="Processing", record=["spikes"])
    
    # 3. Connections
    # Format: [pre_idx, post_idx, weight, delay]
    # Connecting input (0) to all 5 processing neurons with different weights
    conns = []
    for i in range(5):
        weight = 5.0  # Excitatory weight
        delay = 1     # 1ms delay
        conns.append([0, i, weight, delay])
        
    proj = snn.Projection(pre=stim, post=pop_process, connections=conns)
    
    # 4. Build Network
    net = snn.Network("HAR_Sweep_Project")
    net.add(stim, pop_process, proj)
    
    # 5. Connect to Hardware
    # Note: Use 'ExperimentBackendType.SPINNMAN2' if on standard setup, or check lab docs.
    try:
        hw = hardware.SpiNNaker2Chip(eth_ip=S2_IP) 
        print(f"Connected to SpiNNaker 2 at {S2_IP}")
    except Exception as e:
        print(f"Hardware connection failed: {e}")
        print("Simulation will run in default mode (if supported) or fail.")
        return

    # --- PARAMETER SWEEP LOOP ---
    simulation_duration = len(signal) * 10 
    
    for th in thresholds:
        print(f"\n--- Testing Threshold: {th} ---")
        
        # A. Reset Network for new run
        net.reset()
        
        # B. Generate new spikes based on current threshold
        current_input_spikes = delta_encoding(signal, th)
        num_input_spikes = len(current_input_spikes[0])
        print(f"Generated {num_input_spikes} input spikes.")
        
        if num_input_spikes == 0:
            print("Warning: Threshold too high, no spikes generated.")
            results_spike_count.append(0)
            continue
            
        # C. Update Input Population Parameters
        stim.params = current_input_spikes
        
        # D. Run Simulation
        try:
            hw.run(net, simulation_duration)
        except Exception as e:
            print(f"Runtime Error: {e}")
            break
            
        # E. Retrieve Results
        # get_spikes() returns dictionary {neuron_id: [times]}
        output_spikes_dict = pop_process.get_spikes()
        
        # Calculate Total Network Activity (Energy Proxy)
        total_spikes = sum(len(times) for times in output_spikes_dict.values())
        results_spike_count.append(total_spikes)
        print(f"Total Output Spikes (Energy Proxy): {total_spikes}")

    # --- RESULTS VISUALIZATION ---
    print("\n--- Sweep Complete ---")
    print(f"Thresholds: {thresholds}")
    print(f"Activity:   {results_spike_count}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, results_spike_count, marker='o', linestyle='-', color='b')
    plt.title("Parameter Sweep: Threshold vs. Energy Consumption")
    plt.xlabel("Delta Encoding Threshold")
    plt.ylabel("Total Spikes (Energy Proxy)")
    plt.grid(True)
    plt.savefig("har_sweep_result.png")
    print("Graph saved as 'har_sweep_result.png'")

if __name__ == "__main__":
    run_simulation()