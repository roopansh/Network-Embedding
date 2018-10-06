import csv
import numpy as np
from scipy import io
from scipy.sparse import csr_matrix

def load_data_terrorist():
	'''
		Load the graph in the matrix form
	'''
	file = open('data/original/hijackers.csv', 'r')
	data = csv.reader(file)
	table = [row for row in data]
	terrorists = table[0][1:]
	network = table[1:]

	for i, row in enumerate(network):
		network[i] = [float(x) for x in row[1:]]

	# Convert to a undirected network
	for i, row in enumerate(network):
		for j, value in enumerate(row):
			if network[i][j] == 1.0:
				network[j][i] = 1.0

	file.close()

	network2 = np.matmul(network, network)
	network = csr_matrix(network, dtype=int)

	io.savemat('data/sample/hijackers_1.mat', {'adjacency_matrix' : network})
	network2[network2 > 0] = 1	# replace values greater than or equal to 1 with 1 in the second order adjacency matrix
	network = csr_matrix(network2, dtype=int)
	io.savemat('data/sample/hijackers_2.mat', {'adjacency_matrix' : network})


def load_dblp():
	n = 10000
	file = open('data/original/dblp.txt', 'r')
	edge_file = open('data/sample/dblp.edgelist', 'w')
	for line in file.read().split('\n')[:-2]:
		a,b = line.split()
		a = int(a)
		b = int(b)
		if a < n and b < n: #taking all the edges with node values less than 10000
			edge_file.write(line + '\n')
	file.close()
	edge_file.close()

def main():
	load_data_terrorist()
	load_dblp()

if __name__ == '__main__':
	main()
