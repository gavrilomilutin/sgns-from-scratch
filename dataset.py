import re
from collections import Counter
import numpy as np
np.random.seed(42)

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()


def build_vocab(tokens, min_count=1):
    counter = Counter(tokens)

    vocab = {}
    freqs = []

    for word, count in counter.items():
        if count >= min_count:
            vocab[word] = len(vocab)
            freqs.append(count)

    freqs = np.array(freqs, dtype=np.float64)
    return vocab, freqs


def generate_pairs(tokens, vocab, window_size):
    pairs = []

    for i, word in enumerate(tokens):
        if word not in vocab:
            continue

        center_idx = vocab[word]

        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):
            if j != i and tokens[j] in vocab:
                pos_idx = vocab[tokens[j]]
                pairs.append((center_idx, pos_idx))

    return pairs


def negative_sampling_distribution(freqs):

    dist = freqs ** 0.75
    dist /= dist.sum()

    return dist
