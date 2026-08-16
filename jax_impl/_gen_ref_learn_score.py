import os, sys
os.environ.setdefault("JAX_PLATFORMS","cpu")
os.environ.pop("TOKEN_LEARN_SCORE", None)
os.environ.pop("TOKEN_ATTNPROXY", None); os.environ.pop("TOKEN_DILATE", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax.numpy as jnp
from jax_impl.data import _compress_soft_tokens
rng = np.random.default_rng(1234)
B,n,C,D = 2,4,64,8
t = rng.standard_normal((B,n,C,D)).astype(np.float32)
refs = {"t": t}
o_topk = np.asarray(_compress_soft_tokens(jnp.asarray(t), 16, "topk"))
o_dyn  = np.asarray(_compress_soft_tokens(jnp.asarray(t), 32, "dyn"))
refs["topk_k16"] = o_topk
refs["dyn_k32"]  = o_dyn
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ref_learn_score.npz"), **refs)
print("saved ref shapes:", o_topk.shape, o_dyn.shape)
