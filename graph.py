import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from numpy.core.fromnumeric import _all_dispatcher

# Parameters defining shape of the graph
width_dist = 50
depth_dist = 10


def plot(decision_tree):
    root = decision_tree.root
    depth = decision_tree.depth

    segments = []
    labels = []

    # Draws each level of the binary tree
    def bintree_level(node, x, y, width):

        text = f"Room {node['value']}" if node["leaf"] else f"X{node['attribute']} < {node['value']}"
        labels.append((text, x, y))

        if node["left"]:
            xl = x - width
            yl = y - depth_dist
            segments.append([[x, y], [xl, yl]])
            bintree_level(node["left"], xl, yl, width/2)
        if node["right"]: 
            xr = x + width
            yr = y - depth_dist
            segments.append([[x, y], [xr, yr]])
            bintree_level(node["right"], xr, yr, width/2)


    bintree_level(root, 0, 0, width_dist)

    colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segments, linewidths=1, colors=colors, linestyle='solid')

    # Draws the graph
    fig, ax = plt.subplots()
    ax.set_ylim(-(depth * depth_dist + 1), 1)
    ax.set_xlim(-2*width_dist, 2*width_dist)
    ax.add_collection(line_segments)

    for txt,x,y in labels:
        ax.annotate(txt,(x,y))
    
    plt.show()