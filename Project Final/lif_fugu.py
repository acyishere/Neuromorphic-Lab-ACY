"""
Example of LIF fugu neuron.
An input pattern is fed to 5 neurons arranged to produce the AND function.
This is done by setting synaptic weight such that one spike won't trigger
the downstram neuron, but two will.
"""

import os

import matplotlib.pyplot as plt
from spinnaker2 import hardware, helpers, snn

# create stimulus population with 2 spike sources
# Notice that neurons 5-9 reflect neurons 0-4 (flipped pattern).
input_spikes = {0: [1, 2], 1: [1, 2, 3], 2: [2], 3: [3], 4: [1], 5: [1], 6: [3], 7: [2], 8: [1, 2, 3], 9: [1, 2]}
# The AND neurons parallel the arrangement of the two input groups.
# We expect the output to be one cycle delayed.
expected = {0: [2], 1: [4], 2: [3], 3: [4], 4: [2]}

stim = snn.Population(size=10, neuron_model="spike_list", params=input_spikes, name="stim")

# Create 5 AND neurons
# Most of these parameters are the same as default.
neuron_params = {
    "V": 0.0,  # Start at same point as reset.
    "Vspike": 127.0,
    "decay": 0.0,  # 100% decay between cycles. Each time step is independent.
    "Vbias": 0.0,  # Only respond to input spikes, not internal drive.
    "Vreset": 0.0,
    "p": 1.0,  # always fire when we reach threshold
    "delay": 1,
}

pop1 = snn.Population(size=5, neuron_model="lif_fugu", params=neuron_params, name="pop1", record=["spikes"])

# Create connection between stimulus neurons and LIF Fugu neurons.
# Each connection has 4 entries: [pre_index, post_index, weight, delay]
# LIF Fugu uses sending-side delays, so it ignores the last parameter.
# LIF Fugu uses 7 bits for weight: [-127, 127]
# Any weight less than 127 is sufficient to create the AND effect.
# However, we go for roughly 0.75 of range, so 95.
conns = []
conns.append([0, 0, 95, 0])
conns.append([1, 1, 95, 0])
conns.append([2, 2, 95, 0])
conns.append([3, 3, 95, 0])
conns.append([4, 4, 95, 0])
conns.append([5, 0, 95, 0])
conns.append([6, 1, 95, 0])
conns.append([7, 2, 95, 0])
conns.append([8, 3, 95, 0])
conns.append([9, 4, 95, 0])

proj = snn.Projection(pre=stim, post=pop1, connections=conns)

# create a network and add population and projections
net = snn.Network("my network")
net.add(stim, pop1, proj)

# select hardware and run network
hw = hardware.SpiNNaker2Chip(eth_ip=os.environ.get("S2_IP", "192.168.2.52"))
timesteps = 5
hw.run(net, timesteps)

# get results and plot

# get_spikes() returns a dictionary with:
#  - keys: neuron indices
#  - values: lists of spike times per neurons
spike_times = pop1.get_spikes()
print("spikes:  ", spike_times)
print("expected:", expected)
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True)

indices, times = helpers.spike_times_dict_to_arrays(input_spikes)
ax1.plot(times, indices, "|", ms=20)
ax1.set_ylabel("input spikes")
ax1.set_ylim((-0.5, stim.size - 0.5))

indices, times = helpers.spike_times_dict_to_arrays(spike_times)
ax3.plot(times, indices, "|", ms=20)
ax3.set_ylabel("output spikes")
ax3.set_xlabel("time step")
ax3.set_ylim((-0.5, pop1.size - 0.5))
fig.suptitle("lif_neuron")
plt.show()
