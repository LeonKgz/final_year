import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# from matplotlib2tikz import save as tikz_save

sns.axes_style('white')

dir = '../Simulations/Measurements/'
pc_prefix = 'ChannelVariance/paper_check'
# rl_prefix = 'reinforcement_learning'

node_files = [
              # f'{pc_prefix}/adr_no_conf_simulation_results_node_',
              f'{pc_prefix}/no_adr_no_conf_simulation_results_node_',
              f'{pc_prefix}/adr_conf_simulation_results_node_',]
              # f'{rl_prefix}/simulation_results_node_']
gateway_files = [
    f'{pc_prefix}/no_adr_no_conf_gateway_results_',
    f'{pc_prefix}/adr_conf_gateway_results_',
                 # f'{pc_prefix}/adr_no_conf_gateway_results_',
]
                 # f'{rl_prefix}/gateway_results_']
air_interface_files = [
                        f'{pc_prefix}/no_adr_no_conf_air_interface_results_',
                        f'{pc_prefix}/adr_conf_air_interface_results_',]
                       # f'{pc_prefix}/adr_no_conf_air_interface_results_',
                       # f'{rl_prefix}/air_interface_results_']

color = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
         (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
         (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
         (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
         (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(color)):
    r, g, b = color[i]
    color[i] = (r / 255., g / 255., b / 255.)

channel_variance = dict()
channel_var = dict()

payload_sizes = range(5, 55, 5)
path_loss_variances = pd.read_pickle(dir + air_interface_files[0] + '{}'.format(payload_sizes[0])).index.values

for var in path_loss_variances:
    channel_variance[var] = dict()
    channel_var[var] = []
    for p in payload_sizes:
        channel_variance[var][p] = 0
i = 0

j = 0
colors = dict()
for var in path_loss_variances:
    colors[var] = color[j]
    j += 1

num_plots = 3
ax_id = range(0, num_plots)
# Two subplots, the axes array is 1-d
# f, axarr = plt.subplots(2, 2, sharex=True, sharey=False, figsize=(9, 3))
f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))

titles = ["STATIC", "ADR", "DEEP Q"]

for ax_id, air_f, gateway_f, node_f in zip(ax_id, air_interface_files, gateway_files, node_files):
    # ax_id_one = int(ax_id / 2)
    # ax_id_two = int(ax_id % 2)
    # ax = axarr[ax_id_one][ax_id_two]
    # ax.set_title(air_f)
    ax = axarr[ax_id]
    ax.set_title(titles[ax_id])

    # clean variables
    for var in path_loss_variances:
        channel_variance[var] = dict()
        channel_var[var] = []
        for p in payload_sizes:
            channel_variance[var][p] = 0

    for p in payload_sizes:
        air_df = pd.read_pickle(dir + air_f + '{}'.format(p))
        gateway_df = pd.read_pickle(dir + gateway_f + '{}'.format(p))
        node_df = pd.read_pickle(dir + node_f + '{}'.format(p))
        for var, tx, rx in zip(node_df.index.values, node_df.UniquePackets, gateway_df.UniquePacketsReceived):
            channel_variance[var][p] = (rx / tx) * 100

    for p in payload_sizes:
        for var in path_loss_variances:
            channel_var[var].append(channel_variance[var][p])

    for var in path_loss_variances:
        if (ax_id == 2):
            ax.plot(payload_sizes, channel_var[var], marker='o', linestyle='--', label=('$\sigma_{dB}$: ' + str(var) + ' (dB)'), color=colors[var])
        else:
            ax.plot(payload_sizes, channel_var[var], marker='o', linestyle='--', color=colors[var])
        ax.ticklabel_format(axis='y', useOffset=False)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

f.legend(bbox_to_anchor=(1.0, 0.6), loc=2)
f.subplots_adjust(left=0, right=0.02)
f.tight_layout()
plt.xlabel("Payload Size")
plt.ylabel("DER")
plt.show()
f.savefig(f"{os.getcwd()}/Figures/RL/RL_comparison_DER_variance_payload", bbox_inches='tight')

# From inspecting the plots one can see that RL agent results in a higher
# Extraction rate than a plain model without ADR or COnfirmed Messages enabled
# but still performs worse than the most optimal variant with ADR and CONF enabled
