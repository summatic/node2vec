"""
Reference implementation of node2vec for text data.

Author: Hanseok Jo

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016

and github page of original node2vec:
https://github.com/aditya-grover/node2vec
"""

import argparse
import node2vec


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--model_type', type=str, default='w2v',
                        help='Node2vec uses word2vec(fasttext). Default is word2vec.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=2,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    input_path = args.input
    output_path = args.output
    model_type = args.model_type
    dimensions = args.dimensions
    walk_length = args.walk_length
    num_walks = args.num_walks
    window_size = args.window_size
    itr = args.iter
    workers = args.workers
    p = args.p
    q = args.q
    is_weighted = args.weighted
    is_directed = args.directed

    graph = node2vec.Node2Vec(dimensions=dimensions, walk_length=walk_length, num_walks=num_walks,
                              window_size=window_size, itr=itr, workers=workers, p=p, q=q,
                              is_weighted=is_weighted, is_directed=is_directed)
    graph.train(inputs=input_path, model_type=model_type)
    graph.save(output_path=output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
