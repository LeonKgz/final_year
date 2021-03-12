import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

fig, left_ax = plt.subplots()
right_ax = left_ax.twinx()
color = [(31 / 255, 119 / 255, 180 / 255), (158 / 255, 218 / 255, 229 / 255), (140 / 255, 86 / 255, 75 / 255),
         (199 / 255, 199 / 255, 199 / 255), (247 / 255, 182 / 255, 210 / 255)]
left_ax.set_xlabel('payload size [B]')
left_ax.set_ylabel('Energy per Byte [mJ/B]')

####################################################################################################
gateway_df = pd.read_pickle('../Simulations/Measurements/gateway_results_100')
nodes_df = pd.read_pickle('../Simulations/Measurements/simulation_results_node_100')
payload_sizes = gateway_df.index.values
bytes_lost = gateway_df.DLPacketsLost * payload_sizes
print(nodes_df)
print(gateway_df)
energy_per_byte = nodes_df.TxRxEnergy / (gateway_df.UniquePacketsReceived * payload_sizes)
print(energy_per_byte)
retransmitted_bytes = nodes_df.RetransmittedPackets * payload_sizes
wait_time = nodes_df.WaitTimeDC
der = gateway_df.UniquePacketsReceived / nodes_df.UniquePackets

left_ax.plot(payload_sizes, energy_per_byte, marker='o', linestyle='--', lw=1, color=color[1], markersize=10, label='ADR + CONF')
#
# right_ax.plot(payload_sizes, bytes_lost / max(bytes_lost), marker='o', linestyle='--', lw=1, color=color[1],
#               markersize=10, label='Lost DC limit')
# retr_ratio = retransmitted_bytes / max(retransmitted_bytes)
# right_ax.plot(payload_sizes, retr_ratio, marker='o', linestyle='--', lw=1,
#               color=color[1], markersize=10, label='Retrans.')
# rat = (nodes_df.RetransmittedPackets / nodes_df.UniquePackets) * 100
# for x, y, z in zip(payload_sizes, rat, retr_ratio):
#     right_ax.annotate("{0:.2f}%".format(y), xy=(x, z - 0.05), textcoords='data')
####################################################################################################
gateway_df = pd.read_pickle('../Measurements/gateway_results_adr_no_conf_100')
nodes_df = pd.read_pickle('../Measurements/simulation_results_node_adr_no_conf_100')
payload_sizes = gateway_df.index.values
bytes_lost = gateway_df.DLPacketsLost * payload_sizes
print(nodes_df)
print(gateway_df)
energy_per_byte = nodes_df.TxRxEnergy / (gateway_df.UniquePacketsReceived * payload_sizes)
print(energy_per_byte)
retransmitted_bytes = nodes_df.RetransmittedPackets * payload_sizes
wait_time = nodes_df.WaitTimeDC
der = gateway_df.UniquePacketsReceived / nodes_df.UniquePackets

left_ax.plot(payload_sizes, energy_per_byte, marker='o', linestyle='--', lw=1, color=color[2], markersize=10, label='ADR')
#
# right_ax.plot(payload_sizes, bytes_lost / max(bytes_lost), marker='o', linestyle='--', lw=1, color=color[2],
#               markersize=10, label='Lost DC limit')
# retr_ratio = retransmitted_bytes / max(retransmitted_bytes)
# right_ax.plot(payload_sizes, retr_ratio, marker='o', linestyle='--', lw=1,
#               color=color[2], markersize=10, label='Retrans.')
# rat = (nodes_df.RetransmittedPackets / nodes_df.UniquePackets) * 100
# for x, y, z in zip(payload_sizes, rat, retr_ratio):
#     right_ax.annotate("{0:.2f}%".format(y), xy=(x, z - 0.05), textcoords='data')

####################################################################################################
gateway_df = pd.read_pickle('../Measurements/gateway_results_no_adr_no_conf_100')
nodes_df = pd.read_pickle('../Measurements/simulation_results_node_no_adr_no_conf_100')
payload_sizes = gateway_df.index.values
bytes_lost = gateway_df.DLPacketsLost * payload_sizes
print(nodes_df)
print(gateway_df)
energy_per_byte = nodes_df.TxRxEnergy / (gateway_df.UniquePacketsReceived * payload_sizes)
print(energy_per_byte)
retransmitted_bytes = nodes_df.RetransmittedPackets * payload_sizes
wait_time = nodes_df.WaitTimeDC
der = gateway_df.UniquePacketsReceived / nodes_df.UniquePackets

left_ax.plot(payload_sizes, energy_per_byte, marker='o', linestyle='--', lw=1, color=color[3], markersize=10)

# right_ax.plot(payload_sizes, bytes_lost / max(bytes_lost), marker='o', linestyle='--', lw=1, color=color[3],
#               markersize=10, label='Lost DC limit')
# retr_ratio = retransmitted_bytes / max(retransmitted_bytes)
# right_ax.plot(payload_sizes, retr_ratio, marker='o', linestyle='--', lw=1,
#               color=color[3], markersize=10, label='Retrans.')
# rat = (nodes_df.RetransmittedPackets / nodes_df.UniquePackets) * 100
# for x, y, z in zip(payload_sizes, rat, retr_ratio):
#     right_ax.annotate("{0:.2f}%".format(y), xy=(x, z - 0.05), textcoords='data')
left_ax.legend()
plt.show()
