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

   def most_similar(self, word, vocab, top_k=5):
            """
            Return the top_k most cosine-similar words to `word`.
    
            vocab: the word->index dict from build_vocab()
            Returns a list of (word, similarity_score) tuples, sorted by descending similarity.
            """
            if word not in vocab:
                raise KeyError(f"'{word}' not in vocabulary")
    
            # Build reverse mapping once
            idx2word = {idx: w for w, idx in vocab.items()}
    
            # L2-normalise all input vectors for cosine similarity
            vecs = self.W_in
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            vecs_norm = vecs / norms
    
            query = vecs_norm[vocab[word]]
            sims = vecs_norm @ query                  # cosine sim to every word
    
            sims[vocab[word]] = -1.0                  # exclude the query word itself
    
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    
            return [(idx2word[i], float(sims[i])) for i in top_indices]
    
    def analogy(self, pos1, neg1, pos2, vocab, top_k=5):
            """
            Solve: pos1 - neg1 + pos2 ≈ ?
            Example: king - man + woman ≈ queen
    
            vocab: the word->index dict from build_vocab()
            Returns a list of (word, similarity_score) tuples.
            """
            for w in [pos1, neg1, pos2]:
                if w not in vocab:
                    raise KeyError(f"'{w}' not in vocabulary")
    
            idx2word = {idx: w for w, idx in vocab.items()}
    
            vecs = self.W_in
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            vecs_norm = vecs / norms
    
            query = vecs_norm[vocab[pos1]] - vecs_norm[vocab[neg1]] + vecs_norm[vocab[pos2]]
            query /= (np.linalg.norm(query) + 1e-10)
    
            sims = vecs_norm @ query
    
            # Exclude the three input words from results
            for w in [pos1, neg1, pos2]:
                sims[vocab[w]] = -1.0
    
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
    
            return [(idx2word[i], float(sims[i])) for i in top_indices]
