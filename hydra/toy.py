import os, math, time, urllib.request, zipfile
import torch, torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import requests, textwrap, tarfile, io
from tqdm import tqdm
import argparse
import neptune
import random

import argos




def parse_args():
    parser = argparse.ArgumentParser(description="Swarm Experiments on MNIST")
    parser.add_argument("--neptune", action="store_true", help="Log to Neptune?")
    parser.add_argument("--baseline", action="store_true", help="Do vanilla Sinkformer?")
    parser.add_argument("--dataset", type=str, choices=[
            "tinystories",
            "owt",
            "wiki2",
        ], default="tinystories", help="Dataset name"
    )
    parser.add_argument("--enc", type=str, choices=["char", "bpe"], default="char", help="Tokenizer type (char, bpe)")
    parser.add_argument("--device", type=str, choices=["cuda", "gpu", "cpu"], default="gpu")

    parser.add_argument("--batch", type=int, default=64, help="Batch Size")

    parser.add_argument("--argos", type=int, default=-1, help="If >=0, create a 3D (PCA) GIF of the latent semantic flow after this many train steps.")

    # ========== GPT HYPERPARAMS =======

    parser.add_argument("--block", type=int, default=256, help="Context Window Size")
    parser.add_argument("--n_layer", type=int, default=8, help="Number of Layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of Att Heads")
    parser.add_argument("--d", type=int, default=512, help="Model Latent Dim")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout Probability")
    parser.add_argument("--bias", action="store_true", help="Add bias?")
    parser.add_argument("--max_iter_sink", type=int, default=1, help="How many Sinkhorn Steps on Attention Matrix? 1 (default) corresponds to vanilla Softmax.")
    parser.add_argument("--flash", action="store_true", help="(Use Flash Attention (Brrrrr, but no Sinkhorn)")

    # ========== HYDRO HYPERPARAMS =======
    # nu = args.nu,
    # rot_coef = args.rot_coef,

    parser.add_argument("--nu", type=float, default=0.0, help="Viscosity/Laplacian coefficient") # TODO add linear schedule
    parser.add_argument("--rot_coef", type=float, default=0.04, help="Rotator Matrix coefficient")
    parser.add_argument("--rot_init", type=str, choices=[
            "kaiming",
            "block",
            "orthproj",
            "cayley",
        ], default="orthproj", help="Init Strategy for Rotator Matrices"
    )

    # ========== MAGNETO HYPERPARAMS =======
    # TODO FIXME


    return parser.parse_args()


def init_neptune(args):

    with open("../.neptune_tok", "r") as f:
        tok = f.read()

    run = neptune.init_run(
        project="halcyon/hydra",
        api_token=tok,
    )

    # run["parameters/bla"] = args.bla # TODO
    run["parameters/baseline"] = args.baseline

    return run



def main(args):

    if args.neptune:
        run = init_neptune(args)
    else:
        run = {}

    if args.baseline:
        from model import GPT, GPTConfig
    else:
        from hydra_model import GPT, GPTConfig


    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------

    if args.dataset == "wiki2":

        print("Loading WikiText2 via HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(dataset['train']['text'])[:10_000_000]  # 10MB slice
        val_text   = "\n".join(dataset['validation']['text'])

        # Byte-level tokenizer
        chars = sorted(set(train_text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        def encode(s): return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
        def decode(t): return ''.join(itos[int(i)] for i in t)

        train_ids = encode(train_text)
        val_ids   = encode(val_text)

        block_size = args.block
        class CharDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data) - block_size
            def __getitem__(self, idx):
                x = self.data[idx:idx+block_size]
                y = self.data[idx+1:idx+block_size+1]
                return x, y


        trn_data = CharDataset(train_ids)
        val_data = CharDataset(val_ids)

        voc_size = len(stoi)


    elif args.dataset == "tinystories":

        print("Loading TinyStories …")
        ds = load_dataset("roneneldan/TinyStories")  # 17‑tokenised already

        # Concatenate train split into raw string
        raw_text = "\n".join(ds["train"]["text"])
        # Take ~25 MB slice for quick runs (25_000_000 chars)
        raw_text = raw_text[:25_000_000]

        if args.enc == "char":
            # Very simple byte‑level vocab
            chars = sorted(set(raw_text))
            stoi  = {ch:i for i,ch in enumerate(chars)}
            itos  = {i:ch for ch,i in stoi.items()}
            def encode(s): return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)
            def decode(t): return "".join(itos[int(i)] for i in t)

            ids = encode(raw_text)

            voc_size = len(stoi)

        elif args.enc == "bpe":

            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")   # any GPT2‑style BPE
            ids = torch.tensor(tok.encode(raw_text), dtype=torch.long)
            voc_size = tok.vocab_size

        # 90/10 split
        split = int(0.9 * len(ids))
        train_ids = ids[:split]
        val_ids   = ids[split:]

        block_size = args.block

        class TinyDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data) - block_size
            def __getitem__(self, idx):
                x = self.data[idx:idx+block_size]
                y = self.data[idx+1:idx+block_size+1]
                return x, y

        trn_data = TinyDataset(train_ids)
        val_data = TinyDataset(val_ids)

    elif args.dataset == "owt":
        from transformers import AutoTokenizer

        owt = load_dataset("stas/openwebtext-10k", streaming=True, split="train")

        # 1.b  byte‑level or BPE tokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        voc_size = tok.vocab_size

        ids_buffer = []
        for sample in owt:
            ids = tok(sample["text"], return_attention_mask=False,
                truncation=False)["input_ids"]
            ids_buffer.extend(ids)
            # stop at ~30 MB worth of ids  (roughly 30e6 chars ≈ 15e6 tokens)
            if len(ids_buffer) > 15_000_000:
                break

        ids = torch.tensor(ids_buffer, dtype=torch.long)
        print(f"Tokenised {len(ids):,} tokens")

        split = int(0.9 * len(ids))
        train_ids = ids[:split]
        val_ids   = ids[split:]

        block_size = args.block
        class LMChunkDataset(Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self): return len(self.data) - block_size
            def __getitem__(self, idx):
                x = self.data[idx:idx+block_size]
                y = self.data[idx+1:idx+block_size+1]
                return x, y

        trn_data = LMChunkDataset(train_ids)
        val_data = LMChunkDataset(val_ids)

    train_loader = DataLoader(trn_data, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_data, batch_size=args.batch, shuffle=False, drop_last=True)

    # ------------------------------------------------------------------
    # 2. Model instantiation
    # ------------------------------------------------------------------

    device = 'cuda' if torch.cuda.is_available() and not args.device == "cpu" else "cpu"
    print("Using Device:\t", device)
    print(f"Vocab size: {voc_size}")


    cfg = GPTConfig(
        block_size = args.block,
        vocab_size = voc_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.d, # 512,
        dropout    = args.dropout,
        bias       = args.bias,
        max_iter_sink = args.max_iter_sink,
        nu = args.nu,
        rot_coef = args.rot_coef,
        flash = args.flash,
        rot_init = args.rot_init,
    )
    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-1)

    # LR schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=3*len(train_loader))




    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(3):                        # 3 epochs ~ quick demo
        model.train()
        # loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        loop = train_loader
        for step,(x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)

            # --- Prep Visualisation ---
            if step == args.argos:
                argos.panoptes(
                    model,
                    x,
                    streamlines=False,
                    n_components=3,
                )

            logits, loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step()

            ppl = math.exp(min(10, loss.item()))
            # loop.set_postfix({
            #     "loss": f"{loss.item():.3f}",
            #     "ppl": f"{ppl:6.1f}",
            #     "grad": f"{grad_norm:.2f}",
            #     "lr": f"{sched.get_last_lr()[0]:.2e}"
            # })
            print("Loss:\t",loss.item())

            if args.neptune:
                run["train/loss"].append(loss.item())

            if step % 1000 == 999:
                print("Validating ...")
                model.eval(); val_loss=0; n=0
                with torch.no_grad():
                    for vx,vy in val_loader:
                        vx,vy = vx.to(device), vy.to(device)
                        _, l = model(vx,vy); val_loss+=l.item()*len(vx); n+=len(vx)
                val_loss /= n
                ppl = math.exp(min(10,val_loss))
                print(f"ep {epoch} it {step}  train {loss.item():6.3f}   val {val_loss:6.3f}  ppl {ppl:6.1f}  "
                      f"elapsed {time.time()-t0:5.1f}s")

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), f"{ROOT}/hydro_lite_best.pt")
                    print("  > saved checkpoint")
                model.train()

    print("Training done. Best val loss:", best_val)

if __name__ == "__main__":

    args = parse_args()
    main(args)
