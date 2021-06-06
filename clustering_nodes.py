from Global import Config
from bisect import bisect_left

# Group together 'nodes' into multiple clusters around center points
# spread out evenly in the CELL grid
def cluster_nodes(nodes, sector_size):

    # 'dimensions' is the same for vertical and horizontal axis
    dimensions = int(Config.CELL_SIZE / sector_size)
    cluster_centres = []

    xs = [(i+0.5)*sector_size for i in range(dimensions)]
    ys = [(j+0.5)*sector_size for j in range(dimensions)]

    for i in range(dimensions):
        for j in range(dimensions):
            cluster_centres.append((xs[i], ys[j]))

    clusters = {}
    # for key in cluster_centres:
    #     clusters[key] = []

    for n in nodes:
        # Effectively find the right column and row based on node's X and Y coordinates
        x = search_closest(xs, n.location.x)
        y = search_closest(ys, n.location.y)

        if ((x, y) not in clusters):
            clusters[(x, y)] = [n]
        else:
            clusters[(x, y)].append(n)

    return clusters

def cluster_nodes_for_agents(nodes, sector_size, agents):
    # 'dimensions' is the same for vertical and horizontal axis
    dimensions = int(Config.CELL_SIZE / sector_size)
    cluster_centres = []

    xs = [(i + 0.5) * sector_size for i in range(dimensions)]
    ys = [(j + 0.5) * sector_size for j in range(dimensions)]

    for i in range(dimensions):
        for j in range(dimensions):
            cluster_centres.append((xs[i], ys[j]))

    clusters = {}
    # for key in cluster_centres:
    #     clusters[key] = []

    for n in nodes:
        # Effectively find the right column and row based on node's X and Y coordinates
        x = search_closest(xs, n.location.x)
        y = search_closest(ys, n.location.y)

        if ((x, y) not in clusters):
            clusters[(x, y)] = [n]
        else:
            clusters[(x, y)].append(n)

    for agent in agents:
        x = search_closest(xs, agent.location.x)
        y = search_closest(ys, agent.location.y)
        agent.assign_nodes(clusters[(x, y)])

# assuming locs is sorted finds the closest entry in locs to x
def search_closest(locs, x):
    i = bisect_left(locs, x)

    if (not i):
        return locs[0]

    if (i >= len(locs)):
        return locs[i-1]

    closest = locs[i-1] if (abs(locs[i-1] - x) < abs(locs[i] - x)) else locs[i]
    return closest
