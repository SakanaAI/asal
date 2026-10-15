import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches

class GraphBrowser(object):

    def __init__(self, matrix, ax, graph_ax):
        self.G = nx.Graph(matrix)
        self.matrix = matrix
        self.ax = ax
        self.graph_ax = graph_ax
        self.pos = nx.spring_layout(self.G)
        self.rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, fill=False, visible=False)
        self.ax.add_patch(self.rect)
        self.update_graph()

    def update_graph(self):
        self.graph_ax.clear()
        G = nx.Graph(self.matrix)
        nx.draw(G, self.pos, with_labels=True, ax=self.graph_ax)

    def on_click(self, event):
        current_ax = event.inaxes
        cx = event.xdata
        cy = event.ydata

        if current_ax == self.ax:
            row = round(abs(cy))
            col = round(abs(cx))

            # Toggle the value in the adjacency matrix
            self.matrix[row, col] = 1 - self.matrix[row, col]
            self.matrix[col, row] = self.matrix[row, col]

            # Redraw the matrix and graph
            im = self.ax.matshow(self.matrix, cmap=plt.cm.Blues)
            self.update_graph()
            self.ax.add_patch(self.rect)
            plt.draw()

def main(matrix):
    fig, (graph_ax, ax) = plt.subplots(1, 2)
    ax.matshow(matrix, cmap=plt.cm.Blues)
    graph_browser = GraphBrowser(matrix, ax, graph_ax)
    fig.canvas.mpl_connect('button_press_event', graph_browser.on_click)
    plt.show()

    

if __name__ == '__main__':
    num_nodes = 10
    matrix = np.zeros((num_nodes, num_nodes))

    main(matrix)
