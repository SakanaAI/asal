def calc_abundance(substrate, genome_i):
    return substrate['genome'].eq(genome_i).sum().item()

def calc_SAD(substrate, genomes):
    SAD = [] # species abundance distribution
    for i in range(len(genomes)):
        SAD.append(calc_abundance(substrate, i))
    return np.array(SAD)

def get_sub_coverage(substrate, SAD):
    return SAD / (substrate.mem.shape[2] * substrate.mem.shape[3])

def gen_genome_distance_matrix(genomes, neat_config):
    # Creates a similarity matrix between all genomes in a list using NEAT's genome distance
    num_genomes = len(genomes)
    # Initialize the matrix with zeros
    distance_matrix = [[0 for _ in range(num_genomes)] for _ in range(num_genomes)]
    
    # Fill the matrix with distances
    for i in range(num_genomes):
        for j in range(i+1, num_genomes):  # Start from i+1 to avoid redundant calculations
            distance = genomes[i].distance(genomes[j], neat_config)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # The matrix is symmetric
    
    return distance_matrix

def plot_genome_distance_matrix(distance_matrix, title):
    # Plots a heatmap of the genome distance matrix with labels for run name and step number
    fig, ax = plt.subplots()
    cax = ax.imshow(distance_matrix, cmap="viridis", vmin=0)
    fig.colorbar(cax, label="Distance")
    ax.set_title(title)
    plt.show()


def create_knn_net(distance_matrix: NDArray[np.float64], k: int, sub_cov_distr: NDArray[np.float64], ages: NDArray[np.int64]):
    distance_matrix_np = np.array(distance_matrix)
    
    num_nodes = distance_matrix_np.shape[0]
    
    G = nx.Graph()
    
    for i in range(num_nodes):
        if sub_cov_distr[i] > 0:
            G.add_node(i, genome_i=i, substrate_coverage=sub_cov_distr[i], age=ages[i])
    
    for i in G.nodes:
        genome_i = G.nodes[i]['genome_i']
        distances = distance_matrix_np[genome_i, :]
        nearest_indices = np.argsort(distances)
        k_found = 0
        while k_found < k:
            j = nearest_indices[k_found]
            if i != j and G.nodes(j).substrate_coverage > 0:
                G.add_edge(i, j, weight=distances[j])
                k_found += 1
    
    return G

def plot_knn_net(G: nx.Graph, title: str):
    plt.figure(figsize=(10, 8))  # Adjust figure size
    pos = nx.spring_layout(G)  # Compute layout
    
    # Extract node attributes for plotting
    coverages_scaled = np.array([G.nodes[node]['substrate_coverage'] for node in G.nodes]) * 100000  # Scale pop size for visibility
    ages = np.array([G.nodes[node]['age'] for node in G.nodes])

    max_age = max(ages) if max(ages) > 0 else 1  # Ensure max_age is not zero to avoid division by zero\

    age_colors = [plt.cm.summer(age / max_age) for age in ages]  # Normalize ages and map to colormap
    nx.draw(G, pos, with_labels=True, node_size=coverages_scaled, node_color=age_colors, font_size=8, edge_color="gray")
    plt.title(title, fontsize=14)  # Set title with run name and step number
    plt.colorbar(plt.cm.ScalarMappable(cmap='summer'), label='Genome Age')
    plt.show()

# def gen_distance_matrix(self):
#     self.distance_matrix = gen_genome_distance_matrix(self.genomes, self.neat_config)
#     return self.distance_matrix

# def plot_genome_distance_matrix(self, title: str, distance_matrix: NDArray[np.float64] = None):
#     if distance_matrix is None:
#         if self.distance_matrix is None:
#             self.gen_distance_matrix()
#         distance_matrix = self.distance_matrix
#     plot_genome_distance_matrix(distance_matrix, title)

# def plot_knn_net(self, k: int, title: str, substrate: Substrate = None, substrate_coverage: NDArray[np.float64] = None):
#     if not substrate_coverage:
#         if not substrate:
#             raise ValueError("Substrate must be provided if substrate_coverage is not provided")
#         SAD = calc_SAD(substrate, self.genomes)
#         self.SAD = SAD
#         self.substrate_coverage = get_sub_coverage(substrate, SAD)
#     self.distance_matrix = gen_genome_distance_matrix(self.genomes, self.neat_config)
#     self.knn_net = create_knn_net(self.distance_matrix, k, self.substrate_coverage, self.ages)
#     plot_knn_net(self.knn_net, title)