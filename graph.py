import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import math


def plot(decision_tree):

    root = decision_tree.root
    depth = decision_tree.depth

    # In a complete binary tree, the width of the tree is determined by the final level.
    # This will be the max width our tree could ever be
    slots_per_level = 2**depth - 1
    mid_slot = math.floor(slots_per_level/2)
    # TODO utilise the min_separation to prevent node labels from overlapping
    min_separation = 10

    max_width = (slots_per_level * 2) / 2
    depth_step = 5

    segments = []
    labels = []

    # Draws each level of the binary tree
    def bintree_level(node, x, y, slot, depth):

        # The label for the current node
        if node["leaf"]:
            text = f"Room: {node['value']}"
            color = "green"
        else:
            text = f"X{node['attribute']} < {node['value']}"
            color = "blue"
        labels.append((text, x, y, color))

        # Draw the left and right child
        gap = 2**depth
        if node["left"]:
            new_slot = slot - gap
            xl = (new_slot - mid_slot)/slots_per_level * max_width
            yl = y - depth_step
            segments.append([[x, y], [xl, yl]])
            bintree_level(node["left"], xl, yl, new_slot, depth-1)
        if node["right"]: 
            new_slot = slot + gap
            xr = (new_slot - mid_slot)/slots_per_level * max_width
            yr = y - depth_step
            segments.append([[x, y], [xr, yr]])
            bintree_level(node["right"], xr, yr, new_slot, depth-1)


    bintree_level(root, 0, 0, mid_slot, depth)
    colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    line_segments = LineCollection(segments, linewidths=1, colors=colors, linestyle='solid')

    # Draws the graph
    fig, ax = plt.subplots()
    ax.set_ylim(-(depth * depth_step + 5), 5)
    ax.set_xlim(-2.1*max_width, 2.1*max_width)
    ax.add_collection(line_segments)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for txt,x,y, color in labels:
        ax.annotate(txt,(x,y), bbox=dict(boxstyle='square',fc=color, alpha=0.5), ha='center', va='center', fontsize=8)
    
    plt.show()
