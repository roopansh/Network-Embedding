import numpy as np
from sklearn.linear_model import LogisticRegression


def load_embedding(file_path):
    """
        Load the graph embedding in the dictionary
    """

    file = open(file_path, 'r')
    file_contents = file.read().split('\n')
    file.close()

    embedding = {}

    for line in file_contents[1:-1]:
        embedding[int(line.split(' ')[0])] = []

    for line in file_contents[1:-1]:
        data = line.split(' ')
        node = int(data[0])
        for embedding_data in data[1:]:
            embedding[node].append(float(embedding_data))

    return embedding


def load_new_edges(file_path):
    return np.load(file_path).item()


def average_feature(u, v):
    return [sum(item) / 2.0 for item in zip(u, v)]


def hammord_feature(u, v):
    return [x * y for x, y in zip(u, v)]


def l1_distance_feature(u, v):
    return [abs(x - y) for x, y in zip(u, v)]


def l2_distance_feature(u, v):
    return [(x - y) ** 2 for x, y in zip(u, v)]


def calculate_features(new_edges, embedding):
    features = []
    labels = []
    for u in new_edges:
        for v in new_edges[u]:
            x = embedding[u]
            y = embedding[v]

            features.append(average_feature(x, y))
            labels.append(int(new_edges[u][v]))

    return features, labels


def split_data(features, labels):
    p = int(features.__len__() * 0.7)
    return features[:p], labels[:p], features[p + 1:], labels[p + 1:]


def main():
    embedding_files = [['embedding/hijackers/random_walk_1.embeddings',
                        'embedding/hijackers/random_walk_2.embeddings',
                        'embedding/hijackers/bfs_dfs_random_walk.embeddings'],
                       ['embedding/dblp/random_walk_1.embeddings',
                        'embedding/dblp/random_walk_2.embeddings',
                        'embedding/dblp/bfs_dfs_random_walk.embeddings'],
                       ['embedding/higgs/random_walk_1.embeddings',
                        'embedding/higgs/random_walk_2.embeddings',
                        'embedding/higgs/bfs_dfs_random_walk.embeddings']
                       ]

    new_edges = ['data/sample/hijackers_newedges.npy', 'data/sample/dblp_newedges.npy',
                 'data/sample/higgs_newedges.npy']

    for i, new_edge in enumerate(new_edges):
        print(new_edge)
        new_edge = load_new_edges(new_edge)
        for embedding in embedding_files[i]:
            print(embedding)
            embedding = load_embedding(embedding)
            features, labels = calculate_features(new_edge, embedding)

            # Split the edges to training and test data(70% - 30%)
            train_data, train_label, test_data, test_label = split_data(features, labels)

            clf = LogisticRegression(solver='lbfgs').fit(train_data, train_label)

            score = clf.score(test_data, test_label)
            print(score)
            print('\n')
        print('\n')


if __name__ == '__main__':
    main()
