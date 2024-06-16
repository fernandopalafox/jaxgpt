import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

# Parameters
train_val_split = 0.9
block_size = 8  # length of each sequence
batch_size = 4  # number of sequences in each batch
rng_seed = 0

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encoding
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)


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
val_data = data[int(len(data) * train_val_split) :]

# Test block sizes
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
print(train_data[: block_size + 1])
for i in range(block_size):
    context = x[: i + 1]
    target = y[i]
    print(f"Context: {context} -> Target: {target}")


# Define a function to generate a batch of data
def get_batch(split):
    "Generate a batch of data of inputs and targets"
    data = train_data if split == "train" else val_data
    ix = jax.random.randint(
        jax.random.PRNGKey(rng_seed), (batch_size,), 0, len(data) - block_size
    )
    x = jnp.stack([data[i : i + block_size] for i in ix])
    y = jnp.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print(xb.shape, yb.shape)
print("inputs")
print(xb)
print("targents")
print(yb)


# Define a bigram model
class Bigram(nn.Module):
    vocab_size: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.vocab_size)

    def __call__(self, indices, targets=None):
        logits = self.embedding(
            indices
        )  # (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            logits_unrolled = jnp.reshape(logits, (-1, vocab_size))
            targets_unrolled = jnp.reshape(targets, (-1))
            loss = optax.softmax_cross_entropy(
                logits_unrolled, jax.nn.one_hot(targets_unrolled, vocab_size)
            )

        return logits, loss

    def generate(self, context, max_new_tokens, rng_key):
        for _ in range(max_new_tokens):
            logits, loss = self(context)
            final_logits = logits[:, -1, :]
            probs = jax.nn.softmax(final_logits, axis=-1)
            next_token = jax.random.categorical(rng_key, probs)
            context = jnp.append(context, next_token)
        return context


# Initialize the model
m = Bigram(vocab_size)
rng_key, subkey = jax.random.split(jax.random.PRNGKey(rng_seed))
params = m.init(subkey, jnp.zeros((batch_size, block_size), jnp.int32))
logits, loss = m.apply(params, xb, targets=yb)
print(logits.shape, loss)