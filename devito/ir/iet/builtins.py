import cgen as c

from devito.distributed import LEFT, RIGHT
from devito.ir.iet.nodes import Iteration, Element, derive_parameters
from devito.types import Array

__all__ = ['copy', 'gather', 'scatter']


def copy(src, dest, src_offs, dest_offs):
    """
    Construct an IET that copies ``dest.shape`` entries from ``src`` to ``dest``
    starting at the given ``src_offs`` and ``dest_offs``.
    """
    from IPython import embed; embed()
    expr = Expression(c.Assign(1, 2))
    for i in dest.dimensions:
        Iteration = 


def gather(f, dimension, direction):
    """
    Construct an IET that copies ``f``'s halo along given ``dimension``
    and ``direction`` to an :class:`Array`. Return the IET as well as the
    destination :class:`Array`.
    """
    assert f.is_Function
    shape = []
    offsets = []
    for d, i, od, oh, h in zip(f.dimensions, f.symbolic_shape,
                               f._offset_domain, f._offset_halo, f._extent_halo):
        if not d.is_Space:
            continue
        elif d == dimension:
            if direction is LEFT:
                shape.append(h.left)
                offsets.append(oh.left)
            else:
                shape.append(h.right)
                offsets.append(i - od.right)
        else:
            shape.append(i)
            offsets.append(0)
    dest = Array(name='dest', shape=shape, dimensions=f.space_dimensions)
    copy(f, dest, offsets, tuple(0 for i in range(len(shape))))


def scatter(f, shape, offsets):
    """
    Construct an IET that copies data from an :class:`Array` into ``f``'s
    halo along then ``dimension`` and ``direction``. Return the IET as well
    as the source :class:`Array`.
    """
    assert f.is_Function
