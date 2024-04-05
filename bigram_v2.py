import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
max_iters = 5000
eval_interval = 500
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd_dim = 32


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# download the file
# import urllib.request
# url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# filename = 'input.txt'
# urllib.request.urlretrieve(url, filename)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from char to int, and from into to char
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, outputs a list of int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: takes a list of int, outputs a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd_dim, head_size, bias=False)
        self.query = nn.Linear(n_embd_dim, head_size, bias=False)
        self.value = nn.Linear(n_embd_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute the attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd_dim, n_embd_dim)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd_dim, 4 * n_embd_dim),
            nn.ReLU(),
            nn.Linear(4 * n_embd_dim, n_embd_dim), # projection layer
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer Block."""
    def __init__(self, n_embd_dim, n_heads):
        super().__init__()
        head_size = n_embd_dim // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size) # communication
        self.ffwd = FeedForward(n_embd_dim) # computation
        self.ln1 = nn.LayerNorm(n_embd_dim)
        self.ln2 = nn.LayerNorm(n_embd_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BiGramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tk_emb_table = nn.Embedding(vocab_size, n_embd_dim)
        self.position_emb_table = nn.Embedding(block_size, n_embd_dim)
        self.blocks = nn.Sequential(
            Block(n_embd_dim, 4),
            Block(n_embd_dim, 4),
            Block(n_embd_dim, 4),
        )
        self.lm_head = nn.Linear(n_embd_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tk_embd = self.tk_emb_table(idx) # (Batch, Time, Channel)
        pos_embd = self.position_emb_table(torch.arange(T, device=idx.device)) # (Time, Channel)

        x = tk_embd + pos_embd
        x = self.blocks(x)
        logits = self.lm_head(x)

        # logits = torch.permute(logits, (0, 2, 1))
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop the idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, _ = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = BiGramLM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for it in range(max_iters):
    model.train()
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f'Iter {it:4d}, Train loss: {losses["train"]:.2f}, Val loss: {losses["val"]:.2f}')

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, 200)[0].cpu().numpy()))