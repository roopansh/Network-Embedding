cd deepwalk/

# Hijackers Data
echo 'HIJACKERS'
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/hijackers.mat --output ../embedding/hijackers/random_walk_1.embeddings --order 1
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/hijackers.mat --output ../embedding/hijackers/random_walk_2.embeddings --order 2
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/hijackers.mat --output ../embedding/hijackers/bfs_dfs_random_walk.embeddings --bfsdfs True --bfs_p 4 --dfs_q 2

echo

echo 'DBLP'
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/dblp.mat --output ../embedding/dblp/random_walk_1.embeddings --order 1 --workers 5
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/dblp.mat --output ../embedding/dblp/random_walk_2.embeddings --order 2 --workers 5
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/dblp.mat --output ../embedding/dblp/bfs_dfs_random_walk.embeddings --bfsdfs True --bfs_p 4 --dfs_q 2 --workers 5


echo

echo 'HIGGS'
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/higgs.mat --output ../embedding/higgs/random_walk_1.embeddings --order 1 --workers 5
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/higgs.mat --output ../embedding/higgs/random_walk_2.embeddings --order 2 --workers 5
python __main__.py --format mat --matfile-variable-name adjacency_matrix --input ../data/sample/higgs.mat --output ../embedding/higgs/bfs_dfs_random_walk.embeddings --bfsdfs True --bfs_p 4 --dfs_q 2 --workers 5
