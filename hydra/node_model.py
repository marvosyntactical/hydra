"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from sinkhorn import SinkhornDistance

MAX_ITER_SINK = 6

# the following are taken from https://github.com/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb

def ode_solve_euler(z0, t0, t1, f, h_max=0.1):
    """
    Simplest Euler ODE initial value solver
    """
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z


def rk4_step(z0, t0, h, f):
    """
    Performs a single step of the RK4 method

    Parameters:
    - z0: Initial state
    - t0: Initial time
    - h: Step size
    - f: Derivative function of the ODE, f(z, t)

    Returns:
    - z_next: State of the system after step h
    """
    k1 = f(z0, t0)
    k2 = f(z0 + h * k1 / 2, t0 + h / 2)
    k3 = f(z0 + h * k2 / 2, t0 + h / 2)
    k4 = f(z0 + h * k3, t0 + h)

    z_next = z0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return z_next

def ode_solve_rk4(z0, t0, t1, f, h_max=0.05):
    """
    Solves an ODE using the RK4 method

    Parameters:
    - z0: Initial state vector
    - t0: Initial time
    - t1: Final time
    - f: Derivative function of the ODE, f(z, t)

    Returns:
    - z: State of the system at time t1
    """
    # h_max = 0.05  # You can adjust this for a trade-off between accuracy and computational expense
    n_steps = math.ceil((abs(t1 - t0) / h_max).max().item())

    h = (t1 - t0) / n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = rk4_step(z, t, h, f)
        t = t + h
    return z

# ode_solve = ode_solve_rk4
ode_solve = ode_solve_euler


class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        a = grad_outputs
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)


class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        """
        The forward pass computes the ODE solution from the initial state z0 over
        the specified time points t, using the dynamics defined by func.

        Args:
            ctx: Context object that can be used to stash information for backward computation.
            z0: The initial state of the ODE (e.g., initial values of the variables).
            t: A tensor of time points at which the ODE solution is evaluated.
            flat_parameters: Flattened parameters of the ODE function, used for parameter sensitivity analysis.
            func: The function that defines the dynamics of the ODE, dz/dt = f(z, t).

        Returns:
            The solution of the ODE at each time point specified in t.
        """

        # Ensure func is an instance of ODEF, a user-defined class for ODE functions.
        assert isinstance(func, ODEF)

        # Extract batch size and shape of the ODE state from the initial state.
        bs, *z_shape = z0.size()

        # Number of time points.
        time_len = t.size(0)

        # Use torch.no_grad() to prevent tracking of gradients since this is
        # purely for the forward computation of the ODE solution.
        with torch.no_grad():
            # Preallocate a tensor to hold the ODE solution at each time point.
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            # The first solution is the initial condition.
            z[0] = z0

            # Iterate over each time interval and solve the ODE using the specified solver.
            for i_t in range(time_len - 1):
                # Solve the ODE from t[i_t] to t[i_t+1] using the dynamics defined by func.
                # The solver updates z0 to the next time point.
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                # Store the solution.
                z[i_t+1] = z0

        # Save function and tensors for use in the backward pass.
        # Note: It's crucial to clone z to avoid modifying it during the backward pass.
        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)

        # Return the ODE solution at the specified time points.
        return z


    @staticmethod
    def backward(ctx, dLdz):
        """
        Performs the backward pass to compute gradients of the loss with respect to
        the initial conditions, parameters, and time points of the ODE.

        Args:
            ctx: The context object where necessary information has been saved during the forward pass.
            dLdz: The gradient of the loss with respect to the output of the ODE solver.

        Returns:
            Gradients with respect to the initial conditions, time points, parameters, and
            the function defining the ODE dynamics (None for the last because it's typically
            not a directly optimized quantity).
        """

        # Retrieve saved tensors and objects from the forward pass.
        func = ctx.func  # The ODE function. Has arguments x and t.
        t, z, flat_parameters = ctx.saved_tensors  # Time points, solution, and parameters.
        time_len, bs, *z_shape = z.size()  # Unpack dimensions.
        n_dim = np.prod(z_shape)  # Total dimensionality of the state.
        n_params = flat_parameters.size(0)  # Number of parameters.


        def augmented_dynamics(aug_z_i, t_i):
            """
            Defines the dynamics of the augmented system to be solved backwards in time for
            the adjoint method. The augmented system includes the original ODE system
            along with additional equations for the adjoints (gradients) of the state
            variables and parameters.

            Args:
                aug_z_i: A tensor containing the current values of the augmented system,
                        which includes the state of the original ODE system, the adjoint
                        (gradient) of the state, the adjoint of the parameters, and
                        the adjoint of time, for each batch element.
                t_i: The current time tensor for each batch element.

            Returns:
                A tensor that represents the time derivative of the augmented system
                at the current time and state.
            """

            # Split the augmented state into the original state (z_i) and its adjoint (a),
            # ignoring the adjoints of the parameters and time for now.
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # n_dim is the dimensionality of the original state.

            # Unflatten z_i and a to their original shapes.
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)

            # Enable gradient tracking for z_i and t_i, required for computing gradients.
            with torch.set_grad_enabled(True):
                # Detach and require gradients for the inputs to the original ODE function.
                # This is necessary because we're computing gradients with respect to these variables.
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)

                # Evaluate the original ODE function with gradients, also obtaining gradients
                # of the function output with respect to z_i, t_i, and the parameters.
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)

                # Ensure the gradients are in the same device and dtype as z_i;
                # fill with zeros if None (i.e., if not provided by func).
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten func_eval and adfdz for concatenation.
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.reshape(bs, n_dim)

            # Concatenate the time derivative of the original state, the negative adjoint of the state,
            # the negative adjoint of the parameters, and the negative adjoint of time, to form
            # the time derivative of the augmented system.
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        # Prepare for adjoint computation.
        dLdz = dLdz.view(time_len, bs, n_dim)  # Flatten the gradient for convenience.

        # Initialize gradients with respect to state, parameters, and time as zeros.
        adj_z = torch.zeros(bs, n_dim).to(dLdz)  # Gradient w.r.t. state.
        adj_p = torch.zeros(bs, n_params).to(dLdz)  # Gradient w.r.t. parameters.
        adj_t = torch.zeros(time_len, bs, 1).to(dLdz)  # Gradient w.r.t. time.

        # Iterate backward through time points to compute adjoints.
        for i_t in range(time_len-1, 0, -1):
            z_i = z[i_t]  # State at current time.
            t_i = t[i_t]  # Current time.
            f_i = func(z_i, t_i).view(bs, n_dim)  # Evaluate ODE function at current state and time.

            # Direct gradients computation.
            dLdz_i = dLdz[i_t]  # Gradient of loss w.r.t. current state.
            # Gradient of loss w.r.t. current time by chain rule.
            dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints with direct gradients.
            adj_z += dLdz_i  # Adjust state adjoint.
            adj_t[i_t] = adj_t[i_t] - dLdt_i  # Adjust time adjoint.

            # Pack current augmented state for solving backward.
            aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

            # Solve the augmented system backwards to adjust adjoints.
            aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

            # Unpack the adjusted adjoints.
            adj_z[:] = aug_ans[:, n_dim:2*n_dim]
            adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
            adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

        # Adjust the initial time adjoint with direct gradients.
        dLdz_0 = dLdz[0]  # Gradient of loss w.r.t. initial state.
        dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
        adj_z += dLdz_0  # Adjust initial state adjoint.
        adj_t[0] = adj_t[0] - dLdt_0  # Adjust initial time adjoint.

        # Return gradients w.r.t. initial conditions, time, and parameters (None for func as it's not optimized directly).
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func)
        if return_whole_sequence:
            return z
        else:
            return z[-1]

# normal transformer stuff 
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class CausalSelfAttentionAfterLN(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd + 1, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # layer norm
        self.ln_weight = nn.Parameter(torch.ones(config.n_embd + 1))
        self.ln_bias = nn.Parameter(torch.zeros(config.n_embd + 1)) if config.bias else None
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.flash = False
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        x = F.layer_norm(x, self.ln_weight.shape, self.ln_weight, self.ln_bias, 1e-5)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C -= 1 # adjust because time variable was added here

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att_shape = att.shape
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            att = att.view(-1, att_shape[2], att_shape[3])


            sink = SinkhornDistance(1, max_iter=MAX_ITER_SINK)
            # print(f"att.shape={att.shape}")
            sinked_att = sink(att)[0]
            sinked_att = sinked_att * sinked_att.shape[-1]
            # print(f"sinked_att.shape={sinked_att.shape}")
            # print(f"v.shape={v.shape}")
            sinked_att = sinked_att.view(att_shape)
            y = sinked_att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLPafterLN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd + 1, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.ln_weight = nn.Parameter(torch.ones(config.n_embd + 1))
        self.ln_bias = nn.Parameter(torch.zeros(config.n_embd + 1)) if config.bias else None


    def forward(self, x):
        x = F.layer_norm(x, self.ln_weight.shape, self.ln_weight, self.ln_bias, 1e-5)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

def add_time(in_tensor, t):
    batch, time, hidden = in_tensor.shape
    return torch.cat((in_tensor, t.expand(batch, time, 1)), dim=-1)


class SATNetLayer(nn.Module):
    def __init__(self, config):
        import satnet
        super(SATNetLayer, self).__init__()
        n = config.block_size * config.n_embd
        m = 600
        aux = 300
        self.satnet = satnet.SATNet(n, m, aux)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(B, -1)
        is_input = torch.ones_like(x).int()
        o = self.satnet(x, is_input)
        o = o.view(B, T, C)
        return o

class ResSAT(nn.Module):
    def __init__(self, block, config):
        super(ResSAT, self).__init__()
        self.block = block
        self.sat = SATNetLayer(config)

    def forward(self, x):
        h1 = self.block(x)
        h2 = self.sat(h1)
        return h1 + h2


class Block(ODEF):
    def __init__(self, config):
        super(Block, self).__init__()
        self.attn_after_ln = CausalSelfAttentionAfterLN(config)
        self.mlp_after_ln = MLPafterLN(config)

    def forward(self, x, t):
        xt = add_time(x, t)
        h = self.attn_after_ln(xt)
        ht = add_time(h, t)
        o = self.mlp_after_ln(ht)
        return o

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            # h = nn.ModuleList([ResSAT(NeuralODE(Block(config)), config) for _ in range(config.n_layer)]),
            h = nn.ModuleList([NeuralODE(Block(config)) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# I implemented the Transformer as a NODE here
# Im revisiting this file atm, remind me how node works
# IIRC this was really slow compared to the default implementation of the transformer, will benchmark in a sec
# What are advantages and disadvantages of the NODE treatment?
# I want the discretization fidelity to be chosen on the fly, i.e. make the ode approxn be as precise as needed for a given token, how would I do this?

# Advantages of NODE treatment:
# - Continuous time representation allows for more flexible modeling
# - Can potentially capture long-range dependencies better
# - Theoretically more parameter efficient

# Disadvantages:
# - Computationally expensive, especially for long sequences
# - May be overkill for tasks where discrete time steps are sufficient
# - Can be harder to interpret/debug

# To choose discretization fidelity on the fly:
# 1. Implement adaptive step size control in the ODE solver
# 2. Use error estimates to adjust step size dynamically
# 3. Set a tolerance level and adjust fidelity to meet it

# Example implementation:
class AdaptiveNeuralODE(nn.Module):
    def __init__(self, func, rtol=1e-3, atol=1e-3):
        super().__init__()
        self.func = func
        self.rtol = rtol
        self.atol = atol
        
    def forward(self, z0, t):
        solution = ode.odeint_adjoint(
            self.func, 
            z0, 
            t, 
            rtol=self.rtol, 
            atol=self.atol,
            method='dopri5'  # Adaptive Runge-Kutta method
        )
        return solution

# Replace NeuralODE with AdaptiveNeuralODE in the model
# Adjust rtol and atol as needed for precision vs. speed tradeoff
