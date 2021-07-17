from node import Node, plot_graph
from database import Database
import numpy as np
from profile import cosine_distance
### Create adjacency matrix
import matplotlib.pyplot as plt

database = Database()
database.load("database.pkl")
profiles = database.profiles

cutoff = 10
# print(N)
iterations = 1000
# print(cutoff)
# print()


### Node label
names = list(profiles.keys())
nodes = []
i = 0
for p in profiles:
    profile = profiles[p]
    for v in profile.vecs:
        nodes.append(Node(i, [], v))
        i+=1
    #[nodes.append(Node(i, [], v)) for i, v in enumerate(profile.vecs)]

N = len(nodes)
adj_matrix = np.zeros((N, N))
print(N)
for i in range(0, N):
    for j in range(i + 1, N):
        weight = np.e**(1/(cosine_distance(nodes[i].descriptor, nodes[j].descriptor)))
        # print(cosine_distance(nodes[i].descriptor, nodes[j].descriptor))
        if weight < cutoff:
            weight = 0
        else:
            nodes[i].neighbors += tuple([j])
            nodes[j].neighbors += tuple([i])
        adj_matrix[i,j] = weight
        adj_matrix[j,i] = weight

print(adj_matrix.shape)
print(adj_matrix)

random_node_choice = np.random.randint(0, N, size=iterations)
for i in random_node_choice:
    neighbors = nodes[i].neighbors
    neighbor_weight_dict = {}
    print(f'node: {i}; neighbor len: {len(neighbors)}')
    for n in neighbors:
        n_weight = adj_matrix[i][n]
        n_label = nodes[n].label
        if n_label in neighbor_weight_dict:
            neighbor_weight_dict[n_label] += n_weight
        else:
            neighbor_weight_dict[n_label] = n_weight
    
    if len(neighbor_weight_dict) > 0:
        maxx = 0
        maxkey = 0
        for key, value in neighbor_weight_dict.items():
            print(f'value {value}')
            if value > maxx:
                maxx = value
                maxkey = key 
        #max_label = maxkey
        #max(neighbor_weight_dict, key=neighbor_weight_dict.get) 
        print(f'label {maxkey}; max is {maxx}')
        nodes[i].label = maxkey
    
print([nodes[i].label for i in range(len(nodes))])
unique_node = set([])
for node in nodes:
    unique_node.add(node.label)
print(len(unique_node))
fig, ax = plot_graph(nodes, adj_matrix)
plt.show()
