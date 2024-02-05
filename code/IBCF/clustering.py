import sys

sys.path.append("../")

from Word2Vec.clustering import clustering

if __name__ == "__main__":
    clustering("IBCF_emb", "IBCF_TSNE", "IBCF_cluster", "IBCF_cluster", 15)
