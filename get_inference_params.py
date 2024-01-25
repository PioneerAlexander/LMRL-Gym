import jax
from flax.core.frozen_dict import FrozenDict

def flatten_dict(d, parent_key='', sep='_'):
      for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, FrozenDict) or isinstance(v, dict):
          flatten_dict(v, new_key, sep=sep)
        else:
          dim = len(v.shape)
          if dim == 2:
            jax.debug.print("{x}", x=(new_key, (v[0, :5], v.sum())))
          else:
            jax.debug.print("{x}", x=(new_key, (v[:5], v.sum())))

