import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

def draw_dag(edges):

    G = nx.Graph() 
    # G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    G.add_edges_from(edges)
    # G.add_edge(1, 2)
    pos = nx.layout.spiral_layout(G)

    node_sizes = [50]
    M = G.number_of_edges()

    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue',)
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                arrowsize=10, edge_color='red', arrows=True)


    ax = plt.gca()
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    edges = [("abc", 'cdf'), ("abc", "g")]
    draw_dag(edges)