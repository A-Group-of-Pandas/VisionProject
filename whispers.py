from node import Node, plot_graph
from database import Database
import numpy as np
from profile import cosine_distance
### Create adjacency matrix
import matplotlib.pyplot as plt

database = Database()
profiles = database.profiles

cutoff = 1.5
N = len(profiles)
# print(N)
iterations = 50
# print(cutoff)
# print()

adj_matrix = np.zeros((N, N))

### Node label
names = list(profiles.keys())
nodes = [Node(i, [], profiles[names[i]].mean_vector) for i in range(N)]

for i in range(0, N):
    for j in range(i + 1, N):
        weight = 1/(cosine_distance(nodes[i].descriptor, nodes[j].descriptor)**2)
        # print(cosine_distance(nodes[i].descriptor, nodes[j].descriptor))
        if weight < cutoff:
            weight = 0
        else:
            nodes[i].neighbors += tuple([j])
            nodes[j].neighbors += tuple([i])
        adj_matrix[i][j] = weight
        adj_matrix[j][i] = weight

random_node_choice = np.random.randint(0, N, size=iterations)
for i in random_node_choice:
    neighbors = nodes[i].neighbors
    neighbor_weight_dict = {}
    print(i)
    print(neighbors)
    for n in neighbors:
        n_weight = adj_matrix[i][n]
        n_label = nodes[n].label
        if n_label in neighbor_weight_dict:
            neighbor_weight_dict[n_label] += n_weight
        else:
            neighbor_weight_dict[n_label] = n_weight
    
    max_label = max(neighbor_weight_dict, key=neighbor_weight_dict.get)
    nodes[i].label = max_label
    

fig, ax = plot_graph(nodes, adj_matrix)
plt.show()
