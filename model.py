import numpy as np
np.random.seed(42)

def sigmoid(x):
      return 1 / (1 + np.exp(-x))

class SGNS:
    def __init__(self, vocab_size, embedding_dim, negative_samples, lr=0.01):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.K = negative_samples
        self.lr = lr

        # Embeddings initialization
        self.W_in = np.random.randn(vocab_size, embedding_dim) * 0.01 # v_c
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.01 # v_w / v_neg

    def sample_negatives(self, distribution, pos_idx):
        negatives = []

        while len(negatives) < self.K:
            idx = np.random.choice(len(distribution), p=distribution)
            if idx != pos_idx:
                negatives.append(idx)

        return negatives

    def train_step(self, center_idx, pos_idx, distribution):
          
        """
        center_idx: index of center word
        pos_idx: index of positive context word
        distribution: unigram^0.75 distribution for negative sampling
        Computes loss and gradients for a single (center, context) pair.
        """
          
        neg_indices = self.sample_negatives(distribution, pos_idx)

        v_c = self.W_in[center_idx]
        v_w = self.W_out[pos_idx]
        v_neg = self.W_out[neg_indices]

        # Forward pass
        pos_score = np.dot(v_c, v_w)
        pos_sig = sigmoid(pos_score)
        pos_loss = -np.log(pos_sig + 1e-10) # added 1e-10 for numerical stability

      
        neg_loss = 0

        neg_score = np.dot(v_neg, v_c)
        neg_sig = sigmoid(-neg_score)
        neg_loss = -np.sum(np.log(neg_sig + 1e-10)) # added 1e-10 for numerical stability

        loss = pos_loss + neg_loss

        # Gradients
        grad_v = (pos_sig - 1) * v_w + np.sum(neg_sig[:, np.newaxis] * v_neg, axis=0)
        grad_v_w = (pos_sig - 1) * v_c
        grad_v_neg = sigmoid(neg_score)[:, np.newaxis] * v_c

        # Parameter updates
        self.W_in[center_idx] -= self.lr * grad_v
        self.W_out[pos_idx] -= self.lr * grad_v_w
        self.W_out[neg_indices] -= self.lr * grad_v_neg

        return loss

    def train_epoch(self, pairs, neg_dist):
        total_loss = 0

        np.random.shuffle(pairs)

        for center_idx, pos_idx in pairs:
            total_loss += self.train_step(center_idx, pos_idx, neg_dist)

        return total_loss / len(pairs)

