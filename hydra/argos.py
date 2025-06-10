"""
| **B. PCA → 3‑D** | Flatten `(B·T·L, d)` and run `sklearn.decomposition.PCA(n_components=3)`; re‑project each `(B,T,L)` point → `(x,y,z)`; **normalise to unit radius** so we stay on a sphere. | |
| **C. Compute velocity field** | `v = z_{ℓ+1} − z_{ℓ}` (layer axis) → `(B,T,L-1,3)` . | |
| **D. Fit a smooth 3‑D vector field** | Use `scipy.interpolate.RBFInterpolator` (or PyTorch `grid_sample`) on the sphere: feed `(xyz)` coords, target `v`, radial basis `φ(r)=r`.| |
| **E. Dense streamline integration** | Define ∼300 seeding points on sphere (Fibonnacci grid). Integrate `dz/dτ = v̂(x)` for τ∈[0,1] with `scipy.integrate.solve_ivp` (RK45). | |
| **F. Rendering** | For each τ‐step: project the streamline manifolds into a `matplotlib` 3‑D axis (`plot_trisurf`) or use `pyvista` for smoother shading; colour by norm(velocity). | |
| **G. Animation** | Save frames (`matplotlib` → `fig.savefig`) and stitch via `imageio.mimsave('hydra.gif', frames, fps=15)`. | |
"""

import torch, numpy as np, imageio
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch

def hooker(activations):
    def hook(module, inp, out):
        activations.append(out.detach().unsqueeze(2).cpu())
    return hook

def panoptes(
        Z: torch.Tensor # (B,T,L,d) from hook
    ):
    print(f"Casting eyes on latent flow ...")

    # A. capture activations
    # ------------------------------------------------------------

    B,T,L,d = Z.shape
    print(f"Z.shape:\t{Z.shape}")

    Z = Z.reshape(-1, d).numpy()

    # B. PCA to 3-D
    print(f"Fitting PCA ...")
    pca = PCA(n_components=3).fit(Z)
    Z3 = pca.transform(Z)
    Z3 /= np.linalg.norm(Z3, axis=1, keepdims=True)  # on sphere
    Z3 = Z3.reshape(B, T, L, 3)

    # C. velocities
    V  = Z3[:,:,1:] - Z3[:,:,:-1]      # (B,T,L-1,3)
    Zc = Z3[:,:,:-1]                   # centres for velocity vectors
    pts = Zc.reshape(-1,3)             # (N,3)
    vec = V.reshape(-1,3)

    # D. radial‑basis fit
    rbf = RBFInterpolator(pts, vec, kernel='linear', epsilon=0.3)

    # E. streamline seeds (Fibonacci)
    n_seed = 300
    phi = np.pi * (3. - np.sqrt(5.))
    seeds = []
    for i in range(n_seed):
        y = 1 - (i / float(n_seed - 1)) * 2   # y ∈ [1,-1]
        r = np.sqrt(1 - y * y)
        theta = phi * i
        seeds.append([np.cos(theta) * r, y, np.sin(theta) * r])
    seeds = np.array(seeds)

    # integrate each seed
    frames = []
    for step in range(40):
        print(f"Getting Frame {step+1}")
        τ = step / 40
        XYZ = []
        for s in seeds:
            sol = solve_ivp(
                lambda t, z: rbf(z.reshape(1, -1))[0],
                [0, τ], s, t_eval=[τ]
            )
            XYZ.append(sol.y[:,-1])
        XYZ = np.array(XYZ)

        # F. render
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(XYZ[:,0], XYZ[:,1], XYZ[:,2], s=4, c='navy')
        ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
        ax.axis('off')
        fig.tight_layout(pad=0)
        fname = f'frame_{step:03d}.png'
        plt.savefig(fname, dpi=120)
        plt.close(fig)
        frames.append(imageio.imread(fname))

    # G. make gif
    imageio.mimsave('../img/hydra_latent_flow.gif', frames, fps=15)
