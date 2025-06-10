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
from mpl_toolkits.mplot3d import Axes3D

# interactive
import plotly.graph_objects as go
import webbrowser, tempfile


# ------------------------------------------------------------------
def _capture_activations(model, x):
    """Forward hook to collect hidden states at layernorms."""
    activations = []

    def captain(module, inp, out):
        activations.append(out.detach().unsqueeze(2).cpu())

    handles = []
    mutiny = lambda: [h.remove() for h in handles]

    for blocc in model.transformer.h:
        blocc.ln1.register_forward_hook(captain)
        blocc.ln2.register_forward_hook(captain)

    _ = model(x) # forward pass

    mutiny()
    Z = torch.cat(activations, dim=2) # shape (B,T,L,d)

    return Z



# ------------------------------------------------------------------
def panoptes(model,
             x,
             n_components=3,
             streamlines=True,
             fps=15,
             frames=40,
             out_gif='../img/hydra.gif'):
    """
    Render latent trajectories on the unit sphere.

    Parameters
    ----------
    model : nn.Module
        Hydra or vanilla GPT.
    n_components : int
        PCA components; 3 gives 3‑D sphere plot.
    streamlines : bool
        If True: fit RBF vector field and animate streamlines
        If False: just scatter PCA points layer‑by‑layer.
    """
    model.eval()

    Z = _capture_activations(model, x)  # (B,T,L,d)

    B, T, L_, d = Z.shape
    Z_flat = Z.reshape(-1, d).float().numpy()

    # 1. PCA to n_components
    pca = PCA(n_components=n_components).fit(Z_flat)
    Z3 = pca.transform(Z_flat)
    Z3 /= np.linalg.norm(Z3, axis=1, keepdims=True)     # project to sphere
    Z3 = Z3.reshape(B, T, L_, n_components)

    if not streamlines:
        _scatter_traj(Z3, out_gif, fps, frames)
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


def _scatter_traj(Z3, out_gif, fps, frames):
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
                   s=3, c='navy', alpha=0.8)
        ax.set_title(f"")
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.axis('off')
        frame_paths.append(os.path.join(tmpdir, f"layer_{ℓ:03d}.png"))
        plt.savefig(frame_paths[-1], dpi=120, transparent=False)
        plt.close(fig)

    imgs = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_gif, imgs, fps=fps)
    print(f"Saved PCA animation to {out_gif}")



def panoptes_interactive(model, input_ids, layers=None, n_components=3,
                         out_html='hydra_latent_interactive.html',
                         sample=1000):
    """
    Interactive 3‑D layer‑by‑layer latent scatter with a slider.

    Parameters
    ----------
    model, input_ids : same as before
    layers : list[int]  subset of layers to visualise
    sample : int       subsample token points for readability
    """
    model.eval()
    B, T = input_ids.shape
    L = len(model.transformer.h)
    if layers is None:
        layers = list(range(L))

    with torch.no_grad():
        Z = _capture_activations(model, input_ids, layers)  # (B,T,L,d)
    B,T,L,d = Z.shape

    # Sub‑sample tokens for clarity
    if B*T > sample:
        idx = torch.randperm(B*T)[:sample]
        b = idx // T
        t = idx %  T
        Z  = Z[b, t]        # (sample, L, d)
    else:
        Z = Z.reshape(-1, L, d)

    Z_flat = Z.reshape(-1, d).float().numpy()

    pca = PCA(n_components=n_components).fit(Z_flat)
    Z3  = pca.transform(Z_flat)
    Z3 /= np.linalg.norm(Z3, axis=1, keepdims=True)
    Z3  = Z3.reshape(-1, L, 3)          # (N, L, 3)

    # Build Plotly traces
    frames = []
    for ℓ, layer_idx in enumerate(layers):
        pts = Z3[:, ℓ, :]
        trace = go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='markers',
            marker=dict(size=3, color='darkblue', opacity=0.8)
        )
        frames.append(go.Frame(data=[trace], name=f"layer{layer_idx}"))

    # Initial trace (layer 0)
    init_trace = frames[0].data[0]

    layout = go.Layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        width=700, height=700,
        updatemenus=[dict(type='buttons',
            showactive=False,
            y=1,
            x=1.05,
            xanchor='left',
            yanchor='top',
            pad=dict(t=0, r=10),
            buttons=[dict(label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True),
                                 transition=dict(duration=0))])])]
    )

    sliders=[dict(
        steps=[dict(method='animate',
                    args=[[f.name], dict(mode='immediate',
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
                    label=f"Layer {layers[i]}")
               for i, f in enumerate(frames)],
        active=0,
        x=0, y=0, len=1.0
    )]

    fig = go.Figure(data=[init_trace], layout=layout, frames=frames)
    fig.update_layout(sliders=sliders, title="Hydra latent evolution")

    fig.write_html(out_html, auto_open=False)
    print(f"Wrote {out_html}")
    # webbrowser.open('file://' + tempfile.gettempdir()+'/'+out_html
    #   if out_html.startswith(tempfile.gettempdir())
    #                else out_html)
