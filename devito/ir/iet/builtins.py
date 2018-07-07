from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import Iteration, Expression, Callable
from devito.ir.iet.utils import derive_parameters
from devito.types import Array
from devito.tools import is_integer

__all__ = ['copy', 'gather', 'scatter']


def copy(src, dst, name):
    """
    Construct a :class:`Callable` that copies ``dst.shape`` entries
    from ``src`` to ``dst``.
    """
    iet = Expression(DummyEq(dst, src))
    for d, s, i in reversed(list(zip(dst.function.dimensions, dst.shape, dst.indices))):
        if is_integer(i):
            continue
        iet = Iteration(iet, d, s)
    parameters = derive_parameters(iet)
    ret = Callable(name, iet, 'void', parameters, ('static',))
    from IPython import embed; embed()


def gather(src, dimension, direction, fixed_index=None):
    """
    Construct an IET that copies ``src``'s halo along given ``dimension``
    and ``direction`` to an :class:`Array`. Return the IET as well as the
    destination :class:`Array`.
    """
    assert src.is_Function
    fixed_index = fixed_index or {}

    shape = []
    src_indices = []
    dst_indices = []
    for d, i in zip(src.dimensions, src.symbolic_shape):
        if d in fixed_index:
            shape.append(1)
            src_indices.append(fixed_index[d])
            dst_indices.append(0)
        elif d == dimension:
            if direction is LEFT:
                shape.append(src._extent_halo[d].left)
                src_indices.append(d + src._offset_halo[d].left)
                dst_indices.append(d)
            else:
                shape.append(src._extent_halo[d].right)
                src_indices.append(d + i - src._offset_domain[d].right)
                dst_indices.append(d)
        elif d.is_Space:
            shape.append(i)
            src_indices.append(d)
            dst_indices.append(d)
        else:
            assert False
    dst = Array(name='dst', shape=shape, dimensions=src.dimensions)
    return copy(src[src_indices], dst[dst_indices], 'gather_halo_%s' % src.name)


def scatter(dst, dimension, direction, fixed_index=None):
    """
    Construct an IET that copies data from an :class:`Array` into ``dst``'s
    halo along given ``dimension`` and ``direction``. Return the IET as well
    as the source :class:`Array`.
    """
    assert dst.is_Function
