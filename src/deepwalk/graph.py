#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count


class Graph(defaultdict):
	"""Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""
	def __init__(self):
		super(Graph, self).__init__(list)

	def nodes(self):
		return self.keys()

	def adjacency_iter(self):
		return self.iteritems()

	def subgraph(self, nodes={}):
		subgraph = Graph()

		for n in nodes:
			if n in self:
				subgraph[n] = [x for x in self[n] if x in nodes]

		return subgraph

	def make_undirected(self):

		t0 = time()

		for v in self.keys():
			for other in self[v]:
				if v != other:
					self[other].append(v)

		t1 = time()

		self.make_consistent()
		return self

	def make_consistent(self):
		t0 = time()
		for k in iterkeys(self):
			self[k] = list(sorted(set(self[k])))

		t1 = time()

		self.remove_self_loops()

		return self

	def remove_self_loops(self):

		removed = 0
		t0 = time()

		for x in self:
			if x in self[x]:
				self[x].remove(x)
				removed += 1

		t1 = time()

		return self

	def check_self_loops(self):
		for x in self:
			for y in self[x]:
				if x == y:
					return True

		return False

	def has_edge(self, v1, v2):
		if v2 in self[v1] or v1 in self[v2]:
			return True
		return False

	def degree(self, nodes=None):
		if isinstance(nodes, Iterable):
			return {v:len(self[v]) for v in nodes}
		else:
			return len(self[nodes])

	def order(self):
		"Returns the number of nodes in the graph"
		return len(self)

	def number_of_edges(self):
		"Returns the number of nodes in the graph"
		return sum([self.degree(x) for x in self.keys()])/2

	def number_of_nodes(self):
		"Returns the number of nodes in the graph"
		return order()

	def random_walk(self, path_length,order = 1, alpha=0, rand=random.Random(), start=None):
		""" Returns a truncated random walk.

				path_length: Length of the random walk.
				alpha: probability of restarts.
				start: the start node of the random walk.
		"""
		G = self
		if start:
			path = [start]
		else:
			# Sampling is uniform w.r.t V, and not w.r.t E
			path = [rand.choice(list(G.keys()))]

		'''
			For order = 2, double the path lenght and drop the alternate nodes in the walk
		'''
		path_length=path_length*order

		while len(path) < path_length:
			cur = path[-1]
			if len(G[cur]) > 0:
				if rand.random() >= alpha:
					path.append(rand.choice(G[cur]))
				else:
					path.append(path[0])
			else:
				break

		return [str(node) for i, node in enumerate(path) if i%order == 0]

	def node2vec_walk(self, walk_length, start_node, alias_nodes, alias_edges):
		'''
		Simulate a BFS+DFS walk starting from start node.
		'''
		G = self

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			cur_nbrs = sorted(G[cur])
			if len(cur_nbrs) > 0:
				if len(walk) == 1:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
					walk.append(next)
			else:
				break

		return [str(node) for node in walk]

	def get_alias_edge(self, src, dst, p , q):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		G = self

		unnormalized_probs = []

		for dst_nbr in sorted(G[dst]):
			if dst_nbr == src:
				unnormalized_probs.append(1.0/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(1.0)
			else:
				unnormalized_probs.append(1.0/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)


def build_deepwalk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0),order = 1, bfsdfs = False, p = 0, q = 0):
	walks = []

	if not bfsdfs:
		nodes = list(G.nodes())
		for cnt in range(num_paths):
			rand.shuffle(nodes)
			for node in nodes:
				walks.append(G.random_walk(path_length,order=order, rand=rand, alpha=alpha, start=node))
	else :
		alias_nodes, alias_edges = preprocess_transition_probs(G, p, q)
		walks = simulate_walks(G, num_paths, path_length, alias_nodes, alias_edges)
	return walks


def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
											rand=random.Random(0)):
	walks = []

	nodes = list(G.nodes())

	for cnt in range(num_paths):
		rand.shuffle(nodes)
		for node in nodes:
			yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)


def clique(size):
		return from_adjlist(permutations(range(1,size+1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
		"grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
		return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			introw = [int(x) for x in l.strip().split()]
			row = [introw[0]]
			row.extend(set(sorted(introw[1:])))
			adjlist.extend([row])

	return adjlist

def parse_adjacencylist_unchecked(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			adjlist.extend([[int(x) for x in l.strip().split()]])

	return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

	if unchecked:
		parse_func = parse_adjacencylist_unchecked
		convert_func = from_adjlist_unchecked
	else:
		parse_func = parse_adjacencylist
		convert_func = from_adjlist

	adjlist = []

	t0 = time()

	with open(file_) as f:
		with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
			total = 0
			for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
					adjlist.extend(adj_chunk)
					total += len(adj_chunk)

	t1 = time()

	t0 = time()
	G = convert_func(adjlist)
	t1 = time()

	if undirected:
		t0 = time()
		G = G.make_undirected()
		t1 = time()

	return G


def load_edgelist(file_, undirected=True):
	G = Graph()
	with open(file_) as f:
		for l in f:
			x, y = l.strip().split()[:2]
			x = int(x)
			y = int(y)
			G[x].append(y)
			if undirected:
				G[y].append(x)

	G.make_consistent()
	return G


def load_matfile(file_, variable_name="network", undirected=True):
	mat_varables = loadmat(file_)
	mat_matrix = mat_varables[variable_name]

	return from_numpy(mat_matrix, undirected)


def from_networkx(G_input, undirected=True):
		G = Graph()

		for idx, x in enumerate(G_input.nodes_iter()):
				for y in iterkeys(G_input[x]):
						G[x].append(y)

		if undirected:
				G.make_undirected()

		return G


def from_numpy(x, undirected=True):
		G = Graph()

		"""
				Removed the sparse condition check
		"""
		if issparse(x):
				cx = x.tocoo()
				for i,j,v in zip(cx.row, cx.col, cx.data):
						G[i].append(j)
		else:
			raise Exception("Dense matrices not yet supported.")

		if undirected:
				G.make_undirected()

		G.make_consistent()
		return G


def from_adjlist(adjlist):
		G = Graph()

		for row in adjlist:
				node = row[0]
				neighbors = row[1:]
				G[node] = list(sorted(set(neighbors)))

		return G


def from_adjlist_unchecked(adjlist):
		G = Graph()

		for row in adjlist:
				node = row[0]
				neighbors = row[1:]
				G[node] = neighbors

		return G


def preprocess_transition_probs(G, p ,q):
	'''
	Preprocessing of transition probabilities for guiding the random walks.
	'''
	alias_nodes = {}
	for node in G.nodes():
		unnormalized_probs = [1 for nbr in G[node]]
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		alias_nodes[node] = alias_setup(normalized_probs)

	alias_edges = {}

	# for edge in G.edges():
	for node in G:
		for nbr in G[node]:
			alias_edges[(node, nbr)] = G.get_alias_edge(node, nbr, p ,q)
			alias_edges[(nbr, node)] = G.get_alias_edge(nbr, node, p ,q)

	return alias_nodes, alias_edges

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K, dtype=np.int)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

def simulate_walks(G, num_walks, walk_length, alias_nodes, alias_edges):
		'''
		Repeatedly simulate BFS/DFS walks from each node.
		'''
		walks = []
		nodes = list(G.nodes())
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(G.node2vec_walk(walk_length, node, alias_nodes, alias_edges))

		return walks
