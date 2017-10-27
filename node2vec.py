import os
import random
import networkx as nx
from datetime import datetime

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText


class Node2Vec:
    def __init__(self, dimensions=128, walk_length=40, num_walks=10, window_size=2, min_count=0, sg=1, itr=1, workers=8,
                 p=1, q=1, is_weighted=False, is_directed=False):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.min_count = min_count
        self.sg = sg
        self.iter = itr
        self.workers = workers
        self.p = p
        self.q = q
        self.is_weighted = is_weighted
        self.is_directed = is_directed

        self.graph = None
        self.alias_nodes = None
        self.alias_edges = None
        self.emb_model = None

    def _read_graph(self, edgelist):
        """
        Reads the input network in networkx
        """
        graph = self._parse_edgelist(edgelist)

        if not self.is_directed:
            graph = graph.to_undirected()

        self.graph = graph

    def _parse_edgelist(self, edgelist):
        parse = None
        if isinstance(edgelist, list):
            parse = nx.parse_edgelist
        elif isinstance(edgelist, str):
            if os.path.isfile(edgelist):
                parse = nx.read_edgelist
            else:
                raise FileNotFoundError('Edgelist file is not found.')

        if self.is_weighted:
            graph = parse(edgelist, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        else:
            graph = parse(edgelist, nodetype=str, create_using=nx.DiGraph())
            for edge in graph.edges():
                graph[edge[0]][edge[1]]['weight'] = 1
        return graph

    def _node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start node.
        """
        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return [str(i) for i in walk]

    def _simulate_walks(self):
        """
        Repeatedly simulate random walks from each node.
        """
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())
        print('Walk iteration:')
        for walk_iter in range(self.num_walks):
            print(datetime.now(), str(walk_iter+1), '/', str(self.num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._node2vec_walk(start_node=node))

        return walks

    def _get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        graph = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(graph.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight']/p)
            elif graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(graph[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def _preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        graph = self.graph
        is_directed = self.is_directed

        alias_nodes = {}
        for node in graph.nodes():
            unnormalized_probs = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in graph.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in graph.edges():
                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

    def _learn_embeddings(self, walks, model_type):
        if model_type == 'w2v':
            model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=self.min_count,
                             sg=self.sg, workers=self.workers, iter=self.iter)
        elif model_type == 'ft':
            model = FastText(walks, size=self.dimensions, window=self.window_size, min_count=self.min_count,
                             sg=self.sg, workers=self.workers, iter=self.iter)
        else:
            raise TypeError('Model must be \'w2v\' or \'ft\'.')
        return model

    def train(self, inputs, model_type, max_walks_length):
        print(datetime.now(), 'Start reading graph')
        self._read_graph(edgelist=inputs)

        print(datetime.now(), 'Preprocess transition probs')
        self._preprocess_transition_probs()

        print(datetime.now(), 'Simulate walks')
        walks = self._simulate_walks()

        print(datetime.now(), 'Start embedding. # of walk: %d' % len(walks))
        random.shuffle(walks)
        self.emb_model = self._learn_embeddings(walks[:max_walks_length], model_type)
        print(datetime.now(), 'End embedding')

    def save(self, output_path):
        self.emb_model.save(output_path)


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    k = len(probs)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = k*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q


def alias_draw(j, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    k = len(j)

    kk = int(np.floor(np.random.rand()*k))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]
