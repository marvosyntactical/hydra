# run_test.py
import os, torch, gzip, requests, io
from hydro_model import GPT, GPTConfig

# ---------- 1. tiny dataset ----------
URL = "http://mattmahoney.net/dc/text8.zip"
if not os.path.exists("text8.zip"):
    import zipfile, urllib.request, tempfile
    urllib.request.urlretrieve(URL, "data/text8.zip")
    with zipfile.ZipFile("data/text8.zip") as zf:
        with zf.open("data/text8") as f_in, open("data/text8_100k.txt","wb") as f_out:
            f_out.write(f_in.read(100_000))     # first 100k chars

with open("data/text8_100k.txt","r") as f:
    data = f.read().lower()

chars  = sorted(list(set(data)))
stoi   = {ch:i for i,ch in enumerate(chars)}
itos   = {i:ch for ch,i in stoi.items()}
def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return ''.join(itos[int(i)] for i in t)

idx = encode(data)
block = 128
x  = idx[:block].unsqueeze(0)     # (1,T)
y  = idx[1:block+1].unsqueeze(0)

# ---------- 2. model inst ----------
cfg = GPTConfig(
    block_size=block,
    vocab_size=len(chars),
    n_layer=2,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=False
)
model = GPT(cfg)# .cuda()
# x, y = x.cuda(), y.cuda()

# ---------- 3. single fwd/bwd ----------
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
model.train()
loss = model(x, y)[1]
loss.backward()
opt.step()
print("smoke‑test loss:", loss.item())
