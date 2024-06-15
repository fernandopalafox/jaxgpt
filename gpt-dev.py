import jax

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Encoding
char = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda x: ''.join(itos[i] for i in x)

print(encode("hello"))
print(decode(encode("hello")))