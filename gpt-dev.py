import jax
import jax.numpy as jnp

# Parameters
train_val_split = 0.9
block_size = 8  # length of each sequence
rng_seed = 0

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encoding
char = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}


def encode(s):
    return jnp.array([stoi[ch] for ch in s])


def decode(x):
    return "".join(itos[i.item()] for i in x)


print(encode("hello"))
print(decode(encode("hello")))

# # Encode and save
# data = encode(text)
# with open("data/encoded_text.npy", "wb") as f:
#     jnp.save(f, data)

# Load encoded
with open("data/encoded_text.npy", "rb") as f:
    data = jnp.load(f)

# Split into training and validation set
train_data = data[: int(len(data) * train_val_split)]
val_data = data[int(len(data) * train_val_split):]

# Test block sizes
x = train_data[:block_size]
y = train_data[1: block_size + 1]
print(train_data[:block_size+1])
for i in range(block_size):
    context = x[: i + 1]
    target = y[i]
    print(f"Context: {context} -> Target: {target}")
