import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

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


def calculate_features(new_edges, embedding, prop):
    features = []
    labels = []
    for u in new_edges:
        for v in new_edges[u]:
            x = embedding[u]
            y = embedding[v]
            feature = []

            if prop is 'all' or prop is 'avg':
                feature = feature + average_feature(x, y)
            if prop is 'all' or prop is 'ham':
                feature = feature + hammord_feature(x, y)
            if prop is 'all' or prop is 'l1':
                feature = feature + l1_distance_feature(x, y)
            if prop is 'all' or prop is 'l2':
                feature = feature + l2_distance_feature(x, y)

            if feature is not []:
                features.append(feature)
                labels.append(int(new_edges[u][v]))
    return features, labels


def split_data(features, labels, folds=5):
    # The number of elements of each class in a fold
    p = int(features.__len__() / (folds * 2)) + 1

    # divide into positive and negative edges
    features_positive = []
    labels_positive = []
    features_negative = []
    labels_negative = []
    for i in range(features.__len__()):
        if labels[i] == 1:
            features_positive.append(features[i])
            labels_positive.append(labels[i])
        elif labels[i] == 0:
            features_negative.append(features[i])
            labels_negative.append(labels[i])

    random.shuffle(features_positive)
    random.shuffle(features_negative)
    # return the folds, with each fold having equal number of positive & negative samples
    features_to_ret = []
    labels_to_ret = []
    for i in range(folds):
        features_to_ret.append(features_positive[i*p:(i+1)*p] + features_negative[i*p:(i+1)*p])
        labels_to_ret.append(labels_positive[i*p:(i+1)*p] + labels_negative[i*p:(i+1)*p])

    return features_to_ret, labels_to_ret


def link_prediction(features, labels, folds=5):
    accuracy = 0
    auc_roc = 0
    f1_score_micro = 0
    f1_score_macro = 0
    for i in range(folds):
        test_data = features[i]
        test_label = labels[i]
        train_data = []
        train_label = []
        for j in range(folds):
            if i != j:
                train_data = train_data + features[j]
                train_label = train_label + labels[j]

        clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500, n_jobs=-1).fit(train_data, train_label)

        accuracy += clf.score(test_data, test_label)
        proba = clf.predict_proba(test_data)
        auc_roc += roc_auc_score(test_label, [y for x,y in proba])
        predictions = clf.predict(test_data)
        f1_score_micro += f1_score(test_label, predictions, average='micro')
        f1_score_macro += f1_score(test_label, predictions, average='macro')
    print('Accuracy = {0:.5f}'.format(accuracy/folds))
    print('AUC Score = {0:.5f}'.format(auc_roc/folds))
    print('F1 Measure(Micro) = {0:.5f}'.format(f1_score_micro/folds))
    print('F1 Measure(Macro) = {0:.5f}'.format(f1_score_macro/folds))
    print('\n')


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

            # Features taken individually
            print('Taking only Average')
            features, labels = calculate_features(new_edge, embedding, 'avg')
            features, labels = split_data(features, labels, folds=5)
            link_prediction(features, labels, folds=5)

            print('Taking only Hammord')
            features, labels = calculate_features(new_edge, embedding, 'ham')
            features, labels = split_data(features, labels, folds=5)
            link_prediction(features, labels, folds=5)

            print('Taking only L1 Distance')
            features, labels = calculate_features(new_edge, embedding, 'l1')
            features, labels = split_data(features, labels, folds=5)
            link_prediction(features, labels, folds=5)

            print('Taking only L2 Distance')
            features, labels = calculate_features(new_edge, embedding, 'l2')
            features, labels = split_data(features, labels, folds=5)
            link_prediction(features, labels, folds=5)

            # All features taken together
            print('Taking All features together')
            features, labels = calculate_features(new_edge, embedding, 'all')
            features, labels = split_data(features, labels, folds=5)
            link_prediction(features, labels, folds=5)


if __name__ == '__main__':
    main()
