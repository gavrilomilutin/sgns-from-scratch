import numpy as np
from dataset import tokenize, build_vocab, generate_pairs, negative_sampling_distribution
from model import SGNS
np.random.seed(42)

def main():

    with open("sample_text.txt") as f:
        text = f.read()

    tokens = tokenize(text)

    vocab, freqs = build_vocab(tokens)

    pairs = generate_pairs(tokens, vocab, window_size=2)

    neg_dist = negative_sampling_distribution(freqs)

    model = SGNS(
        vocab_size=len(vocab),
        embedding_dim=50,
        negative_samples=10,
        lr=0.005
    )

    epochs = 50

    for epoch in range(epochs):

        loss = model.train_epoch(pairs, neg_dist)

        print(f"Epoch {epoch+1}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
