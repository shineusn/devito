from itertools import product

from sympy import S
import numpy as np

from devito.dimension import IncrDimension
from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import (ArrayCast, Callable, Conditional, Expression,
                                 Iteration, List)
from devito.ir.iet.utils import derive_parameters
from devito.types import Array, Scalar
from devito.tools import is_integer

__all__ = ['copy', 'halo_exchange']


def copy(src, fixed):
    """
    Construct a :class:`Callable` copying an arbitrary convex region of ``src``
    into a contiguous :class:`Array`.
    """
    src_indices = []
    dst_indices = []
    dst_shape = []
    dst_dimensions = []
    for d in src.dimensions:
        dst_d = IncrDimension(d, S.Zero, S.One, name='dst_%s' % d)
        dst_dimensions.append(dst_d)
        if d in fixed:
            src_indices.append(fixed[d])
            dst_indices.append(0)
            dst_shape.append(1)
        else:
            src_indices.append(d + Scalar(name='o%s' % d, dtype=np.int32))
            dst_indices.append(dst_d)
            dst_shape.append(dst_d)
    dst = Array(name='dst', shape=dst_shape, dimensions=dst_dimensions)

    iet = Expression(DummyEq(dst[dst_indices], src[src_indices]))
    for sd, dd, s in reversed(list(zip(src.dimensions, dst.dimensions, dst.shape))):
        if is_integer(s):
            continue
        iet = Iteration(iet, sd, s.symbolic_size, uindices=dd)
    iet = List(body=[ArrayCast(src), ArrayCast(dst), iet])
    parameters = derive_parameters(iet)
    return Callable('copy', iet, 'void', parameters, ('static',))


def halo_exchange(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    assert f.is_Function
    from IPython import embed; embed()

    # Compute send/recv array buffers
    buffers = []
    for d0, i in product(f.dimensions, [LEFT, RIGHT]):
        if d0 in fixed:
            continue
        shape = [2]  # 2 -- 1 for send buffer, 1 for recv buffer
        offsets = []
        for d1, s in zip(f.dimensions, f.symbolic_shape):
            if d1 in fixed:
                shape.append(1)
                offsets.append(fixed[d1])
            elif d0 is d1:
                if i is LEFT:
                    shape.append(f._extent_halo[d0].left)
                    offsets.append(f._offset_halo[d0].left)
                else:
                    shape.append(f._extent_halo[d0].right)
                    offsets.append(i - f._offset_domain[d0].right)
            else:
                shape.append(s)
                offsets.append(0)
            buffers.append(Array(name='buffer_%s%s' % (d, i.name[0])), shape=shape)

    for d in f.dimensions:
        for i in [LEFT, RIGHT]:

            mask = Scalar(name='m_%s%s' % (d, i.name[0]), dtype=np.int32)
            cond = Conditional(mask, ...)

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
    return copy(src, fixed_index)
