# sgns-from-scratch
Word2Vec (Skip-Gram with Negative Sampling) implemented from scratch in NumPy.

Explained in more detail here: https://arxiv.org/pdf/1402.3722

Why Unigram 3/4 distribution and not Uniform?
Because Uniform distribution is "too flat", rare words will appear just as much as the frequent ones, so the model spends too much time pushing down rare words.
If we sample from raw Unigram, most negatives will be stopwords, so the model will mainly learn to push away words like "the", while rare words barely get trained.
Raising it to the power of 0.75 flattens the distribution just enough for the frequent words to be less dominant and rare words to be relatively more likely to appear, but not too much, so the noise stays realistic. Exact number 0.75 proved to be best empirically, there is no theoretical proof that it's optimal.

Different initialization could yield better results.

Short sample text is provided for testing purposes.
