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
    
    
    
"""
argos.py  – latent‑flow visualisation for Hydra / vanilla GPT

Usage
-----
from argos import panoptes
panoptes(model, input_ids, layers=[0,3,7], streamlines=True)

Dependencies: torch, numpy, sklearn, scipy, matplotlib, imageio
"""

import torch, numpy as np, imageio, os, tempfile
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

# ------------------------------------------------------------------
def _capture_activations(model, x, layers):
    """Forward hook to collect hidden states at selected layers."""
    acts = []

    def hook(_, __, output):
        acts.append(output.detach().cpu())

    handles = []
    for idx in layers:
        h = model.transformer.h[idx].register_forward_hook(hook)
        handles.append(h)
    _ = model(x)          # forward pass
    for h in handles: h.remove()

    # acts list: len = len(layers); each (B,T,d)
    return torch.stack(acts, dim=2)   # (B,T,L,d)


# ------------------------------------------------------------------
def panoptes(model,
             input_ids,
             layers=None,
             n_components=3,
             streamlines=True,
             fps=15,
             frames=40,
             out_gif='hydra.gif'):
    """
    Render latent trajectories on the unit sphere.

    Parameters
    ----------
    model : nn.Module
        Hydra or vanilla GPT.
    input_ids : torch.LongTensor (B,T)
        Token IDs to feed.
    layers : list[int] or None
        Which layers to sample. None -> every layer.
    n_components : int
        PCA components; 3 gives 3‑D sphere plot.
    streamlines : bool
        If True: fit RBF vector field and animate streamlines
        If False: just scatter PCA points layer‑by‑layer.
    """
    model.eval()
    B, T = input_ids.shape
    L = len(model.transformer.h)

    if layers is None:
        layers = list(range(L))
    Z = _capture_activations(model, input_ids, layers)  # (B,T,L,d)

    B, T, L_, d = Z.shape
    Z_flat = Z.reshape(-1, d).float().numpy()

    # 1. PCA to n_components
    pca = PCA(n_components=n_components).fit(Z_flat)
    Z3 = pca.transform(Z_flat)
    Z3 /= np.linalg.norm(Z3, axis=1, keepdims=True)     # project to sphere
    Z3 = Z3.reshape(B, T, L_, n_components)

    if not streamlines:
        _scatter_traj(Z3, layers, out_gif, fps, frames)
        return

    # 2. velocities along layer axis
    V = Z3[:, :, 1:] - Z3[:, :, :-1]                    # (B,T,L-1,3)
    Zc = Z3[:, :, :-1]                                  # centres
    pts = Zc.reshape(-1, n_components)
    vec = V.reshape(-1, n_components)

    # 3. Fit RBF vector field
    rbf = RBFInterpolator(pts, vec, kernel='linear', epsilon=0.3)

    # 4. Fibonacci seeds on sphere
    n_seed = 300
    phi = np.pi * (3. - np.sqrt(5.))
    seeds = []
    for i in range(n_seed):
        y = 1 - (i / float(n_seed - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        seeds.append([np.cos(theta) * r, y, np.sin(theta) * r])
    seeds = np.array(seeds)

    tmpdir = tempfile.mkdtemp()
    frame_paths = []

    for step in range(frames):
        τ = step / (frames - 1)
        XYZ = []
        for s in seeds:
            sol = solve_ivp(lambda t, z: rbf(z.reshape(1, -1))[0],
                            [0, τ], s, t_eval=[τ], rtol=1e-4, atol=1e-6)
            XYZ.append(sol.y[:, -1])
        XYZ = np.array(XYZ)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2],
                   s=3, c='steelblue', alpha=0.9)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.axis('off')
        fig.tight_layout(pad=0)
        fname = os.path.join(tmpdir, f"frame_{step:03d}.png")
        plt.savefig(fname, dpi=120, transparent=True)
        plt.close(fig)
        frame_paths.append(fname)

    imgs = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, imgs, fps=fps)
    print(f"Saved animation to {out_gif}")


def _scatter_traj(Z3, layers, out_gif, fps, frames):
    """
    Simple PCA scatter: show token points layer‑by‑layer.
    """
    B, T, L, _ = Z3.shape
    tmpdir = tempfile.mkdtemp()
    frame_paths = []

    for ℓ in range(L):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        pts = Z3[:, :, ℓ, :].reshape(-1, 3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=3, c='darkorange', alpha=0.8)
        ax.set_title(f"Layer {layers[ℓ]}")
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.axis('off')
        frame_paths.append(os.path.join(tmpdir, f"layer_{ℓ:03d}.png"))
        plt.savefig(frame_paths[-1], dpi=120, transparent=True)
        plt.close(fig)

    imgs = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, imgs, fps=fps)
    print(f"Saved PCA animation to {out_gif}")

