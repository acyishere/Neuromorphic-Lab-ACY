"""
Example of Leaky Integrate&Fire neuron where synaptic events directly increase
the membrane voltage.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from spinnaker2 import hardware, helpers, snn

# Global setup (redundant if different)
logger = logging.getLogger(__name__)
# logger.setup(level="debug", persistency="volatile")

multi_node_board = True  # Set to False for single-node board

if multi_node_board:
    STM_IP = os.environ.get("STM_IP", "192.168.1.2")
    hw = hardware.SpiNNcloud48NodeBoard(stm_ip=STM_IP)
else:
    S2_IP = os.environ.get("S2_IP", "192.168.1.17")
    hw = hardware.SpiNNaker2Chip(eth_ip=S2_IP)

# create stimulus population with 2 spike sources
input_spikes = {0: [1, 4, 9, 11], 1: [20, 30]}
stim = snn.Population(size=2, neuron_model="spike_list", params=input_spikes, name="stim")

# create LIF population with 1 neuron
neuron_params = {
    "threshold": 10.0,
    "alpha_decay": 0.9,
    "i_offset": 0.0,
    "v_init": 0.0,
    "v_reset": 0.0,  # only used for reset="reset_to_v_reset"
    "reset": "reset_by_subtraction",  # "reset_by_subtraction" or "reset_to_v_reset"
}

pop1 = snn.Population(size=1, neuron_model="lif", params=neuron_params, name="pop1", record=["spikes", "v"])

# create connection between stimulus neurons and LIF neuron
# each connection has 4 entries: [pre_index, post_index, weight, delay]
# for connections to a `lif` population:
#  - weight: integer in range [-15, 15]
#  - delay: integer in range [0, 7]. Actual delay on the hardware is: delay+1
conns = []
conns.append([0, 0, 4, 1])  # excitatory synapse with weight 4 and delay 1
conns.append([1, 0, -3, 2])  # inhibitory synapse with weight -3 and delay 1
proj = snn.Projection(pre=stim, post=pop1, connections=conns)

# create a network and add population and projections
net = snn.Network("my network")
net.add(stim, pop1, proj)
timesteps = 50
hw.run(net, timesteps)

# get results and plot

# get_spikes() returns a dictionary with:
#  - keys: neuron indices
#  - values: lists of spike times per neurons
spike_times = pop1.get_spikes()

# get_voltages() returns a dictionary with:
#  - keys: neuron indices
#  - values: numpy arrays with 1 float value per timestep per neuron
voltages = pop1.get_voltages()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

# Input spikes
indices, times = helpers.spike_times_dict_to_arrays(input_spikes)
ax1.plot(times, indices, "|", ms=20)
ax1.set_ylabel("input spikes")
ax1.set_ylim((-0.5, stim.size - 0.5))

# Voltage trace
voltages = pop1.get_voltages()
times = np.arange(timesteps)
ax2.plot(times, voltages[0], label="Neuron 0")
ax2.axhline(neuron_params["threshold"], ls="--", c="0.5", label="threshold")
ax2.axhline(0, ls="-", c="0.8", zorder=0)
ax2.set_xlim(0, timesteps)
ax2.set_ylabel("voltage")

# Output spikes
indices, times = helpers.spike_times_dict_to_arrays(spike_times)
ax3.plot(times, indices, "|", ms=20)
ax3.set_ylabel("output spikes")
ax3.set_xlabel("time step")
ax3.set_ylim((-0.5, pop1.size - 0.5))
fig.suptitle("lif_neuron")
plt.show()
# Uncomment the following lines if you want to save the plot:
# filename = "lif_output.png"
# fig.savefig(filename)
