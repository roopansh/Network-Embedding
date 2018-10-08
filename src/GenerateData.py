import csv
from random import random

import numpy as np
from scipy import io
from scipy.sparse import csr_matrix


# Generate new Edges and the graph at time T-1
def generate_network(network, num_of_nodes):
    # Network at time t
    network_t = [[0 for y in range(num_of_nodes)] for x in range(num_of_nodes)]

    # New edges(YES & NO) at time t+1
    new_edges = {x: {} for x in range(num_of_nodes)}

    threshhold = 0.60

    # YES links
    count = 0
    for i, row in enumerate(network):
        for j, value in enumerate(row):
            if value == 1:  # if yes edge
                # check if it has more than 1 edge
                neighbours = 0
                # check for i
                for u in network_t[i]:
                    if u == 1:
                        neighbours = neighbours + 1
                    if neighbours > 1:
                        break
                if neighbours <= 1:
                    # Add to the graph at time t
                    network_t[i][j] = 1
                    network_t[j][i] = 1
                    continue

                neighbours = 0
                # check for j
                for u in network_t[j]:
                    if u == 1:
                        neighbours = neighbours + 1
                    if neighbours > 1:
                        break
                if neighbours <= 1:
                    # Add to the graph at time t
                    network_t[i][j] = 1
                    network_t[j][i] = 1
                    continue

                # check for j
                if random() <= threshhold:
                    # Add to the graph at time t
                    network_t[i][j] = 1
                    network_t[j][i] = 1
                else:
                    # Add to the new YES edges
                    new_edges[i][j] = 1
                    new_edges[j][i] = 1
                    count += 1

    print('The number of positive edges in the testing dataset = {}'.format(count))

    # NO links
    while count > 0:
        for i, row in enumerate(network):
            for j, value in enumerate(row):
                # If the edge is not present in the original graph at time t+1
                # and has not been added to the new edges yet
                if value == 0 and (i not in new_edges or j not in new_edges[i]):
                    neighbours = False
                    # check for i
                    for u in network_t[i]:
                        if u == 1:
                            neighbours = True
                        if neighbours:
                            break
                    if not neighbours:
                        continue

                    neighbours = False
                    # check for j
                    for u in network_t[j]:
                        if u == 1:
                            neighbours = True
                        if neighbours:
                            break
                    if not neighbours:
                        continue

                    # Add to the new NO edges
                    if random() >= threshhold:
                        new_edges[i][j] = 0
                        new_edges[j][i] = 0
                        count -= 1
                if count <= 0:
                    break
            if count <= 0:
                break

    # Return the generated network of time T and the new edges(YES & NO)
    # which will be added to network at time T to generate the network at time T+1
    return network_t, new_edges


def load_data_terrorist():
    """
        Load the graph in the matrix form
    """
    print('HIJACKERS DATASET')
    file = open('data/original/hijackers.csv', 'r')
    data = csv.reader(file)
    table = [row for row in data]
    terrorists = table[0][1:]
    network = table[1:]

    num_of_nodes = terrorists.__len__()

    for i, row in enumerate(network):
        network[i] = [int(x) for x in row[1:]]

    # Convert to a undirected network
    for i, row in enumerate(network):
        for j, value in enumerate(row):
            if network[i][j] == 1:
                network[j][i] = 1

    file.close()

    network, new_edges = generate_network(network, num_of_nodes)

    network = csr_matrix(network, dtype=int)
    io.savemat('data/sample/hijackers.mat', {'adjacency_matrix': network})

    np.save('data/sample/hijackers_newedges.npy', new_edges)


def load_dblp():
    print('DBLP DATASET')
    n = 10000
    file = open('data/original/dblp.txt', 'r')
    network = [[0 for x in range(n)] for x in range(n)]
    for line in file.read().split('\n')[:-2]:
        a, b = line.split()
        a = int(a)
        b = int(b)
        if a < n and b < n:  # taking all the edges with node values less than 10000
            network[a][b] = 1
            network[b][a] = 1
    file.close()

    network, new_edges = generate_network(network, n)
    network = csr_matrix(network, dtype=int)
    io.savemat('data/sample/dblp.mat', {'adjacency_matrix': network})

    np.save('data/sample/dblp_newedges.npy', new_edges)


# Load data and generate the network
def load_higgs():
    print('HIGGS DATASET')
    num_of_nodes = 1000
    network = [[0 for y in range(num_of_nodes)] for x in range(num_of_nodes)]

    file = open('data/original/higgs.txt', 'r')
    file_content = file.readlines()
    for row in file_content:
        nodes = row.split()
        node1 = int(nodes[0]) - 1
        node2 = int(nodes[1]) - 1
        if node1 < num_of_nodes and node2 < num_of_nodes:
            network[node1][node2] = 1
            network[node2][node1] = 1

    file.close()

    network, new_edges = generate_network(network, num_of_nodes)
    network = csr_matrix(network, dtype=int)
    io.savemat('data/sample/higgs.mat', {'adjacency_matrix': network})

    np.save('data/sample/higgs_newedges.npy', new_edges)


def main():
    load_data_terrorist()
    load_dblp()
    load_higgs()


if __name__ == '__main__':
    main()
