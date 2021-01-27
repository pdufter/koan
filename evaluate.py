import argparse
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_distances
import torch
from collections import Counter


def get_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cosine_distances(X, Y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", default=None, type=str, required=True, help="")
    parser.add_argument("--trafo", action="store_true", help="")
    args = parser.parse_args()
    with open(args.vectors) as fp:
        words = []
        vectors = []
        for line in fp:
            line = line.strip().split(" ")
            word, vector = line[0], line[1:]
            words.append(word)
            vectors.append(np.array([float(x) for x in vector]))
    vectors = np.array(vectors)
    word2index = {w: i for (i, w) in enumerate(words)}
    real_vocab = set([x for x in word2index if not x.startswith("::")])
    fake_vocab = set([x[2:] for x in word2index if x.startswith("::")])
    both = real_vocab & fake_vocab
    print(real_vocab - fake_vocab)
    print(fake_vocab - real_vocab)
    # TODO strange bug that disappointeth and enterprise are not in there?
    real = sorted(both)
    fake = sorted(["::" + x for x in both])
    for real_w, fake_w in zip(real, fake):
        if real_w != fake_w[2:]:
            raise ValueError("Token inconsistency.")
    real_indices = np.array([word2index[x] for x in real])
    fake_indices = np.array([word2index[x] for x in fake])
    def get_precision(vectors_x, vectors_y):
        vectors_real = vectors_x[real_indices]
        vectors_fake = vectors_y[fake_indices]
        if args.trafo:
            W = np.linalg.inv(vectors_real.transpose().dot(vectors_real)).dot(vectors_real.transpose().dot(vectors_fake))
            vectors_real = vectors_real.dot(W)
        dist = get_distances(vectors_real, vectors_fake)
        if dist.shape[0] != dist.shape[1]:
            print("Number of words is different?")
        # get different p@k
        nns = np.argsort(dist, axis=1)[:, :10]
        gt = np.arange(dist.shape[0]).reshape(-1, 1)
        p = {}
        for considern in [1, 5, 10]:
            hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
            p[considern] = hits1 / dist.shape[0]
        nns = np.argsort(dist, axis=0)[:10, :].transpose()
        gt = np.arange(dist.shape[0]).reshape(-1, 1)
        pinv = {}
        for considern in [1, 5, 10]:
            hits1 = ((nns[:, :considern] == gt).sum(axis=1) > 0).sum()
            pinv[considern] = hits1 / dist.shape[0]
        return p, pinv

    def get_my_distance(vectors_x, vectors_y):
        vectors_real = vectors_x[real_indices]
        vectors_fake = vectors_y[fake_indices]
        dist = get_distances(vectors_real, vectors_fake)
        if dist.shape[0] != dist.shape[1]:
            print("Number of words is different?")
        # get different p@k
        nns = np.argsort(dist, axis=1)
        return dist, nns

    def mean_rank(nns):
        # Counter(np.where(nns == np.arange(dist.shape[0]).reshape(-1, 1))[1]).most_common(10)
        return np.where(nns == np.arange(dist.shape[0]).reshape(-1, 1))[1].mean()

    def get_details(queries, nns, real, fake):
        for query in queries:
            print("{} - ".format(query), end="")
            for _, nn in zip(range(10), nns[real.index(query)]):
                print("{}".format(fake[nn]), end="|")
            print()

    dist, nns = get_my_distance(vectors, vectors)
    print("MEAN RANK: {}".format(mean_rank(nns)))
    get_details(["the", ".", ",", ":", "God", "LORD", "run", "go"], nns, real, fake)
    print(get_precision(vectors, vectors))
    import ipdb;ipdb.set_trace()
    # x[list(vocab).index("the")].dot(x[list(vocab).index("::the")])
    # vectors[words.index("saith")].dot(vectors[words.index("saith")])
    # norm[words.index("saith")].dot(norm[words.index("saith")])
    # norm = vectors / np.linalg.norm(vectors, axis=1).reshape(-1, 1)

if __name__ == '__main__':
    main()
