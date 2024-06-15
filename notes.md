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
  - 