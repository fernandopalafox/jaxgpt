# Resources
https://www.youtube.com/watch?v=eMlx5fFNoYc

# Notes
- A well-trained attention block tells you what you need to add to move the embedding of a word in the direction that's consistent with the meaning given by the words close to it
- Self attention: 
  - Create three vectors from each of the encoder's iput vectors
  - Query, key, and value
  - Score is the dot product of the query and the key. So, if interested in the score for word at position 1, the first score is computed by the dot product of q1 and k1. Second score is the dot product q1 and k2. 
  - In multi-headed attention, we keep separate Q/K/V matrices for each head. 
  - Afterwards, we're left with one matrix per attention head. We train an additional weight matrix that puts them all together. 
  - Positional encoding: A vector added to the token embeddings that adds information about the order of the word
- Just implemented a very simple attention mechanism where, for every batch, for every element of the batch, I compute the average of all the embeddings up to and including the element. I think that in the real thing. The weight matrix is currently a lower triangular matrix where each row is scaled such that it computes an average, but I think that in the real thing we will also train these weights. Not sure where self-attention falls into all this yet. 