from collections import OrderedDict
from itertools import product
from operator import mul

from sympy import S
import numpy as np

from devito.dimension import DefaultDimension, Dimension
from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import (ArrayCast, Call, Callable, Conditional, Expression,
                                 Iteration, List)
from devito.ir.iet.utils import derive_parameters
from devito.types import Array, Scalar, OWNED, HALO
from devito.tools import is_integer, numpy_to_mpitypes

__all__ = ['copy', 'mpi_exchange']


def copy(f, swap=False):
    """
    Construct a :class:`Callable` capable of copying: ::

        * an arbitrary convex region of ``f`` into a contiguous :class:`Array`, OR
        * if ``swap=True``, a contiguous :class:`Array` into an arbitrary convex
          region of ``f``.
    """
    src_indices, dst_indices = [], []
    src_dimensions, dst_dimensions = [], []
    for d in f.dimensions:
        dst_dimensions.append(Dimension(name='dst_%s' % d.root))
        src_dimensions.append(Dimension(name='src_%s' % d.root))
        src_indices.append(d.root + Scalar(name='o%s' % d.root, dtype=np.int32))
        dst_indices.append(d.root)
    src = Array(name='src', dimensions=src_dimensions, dtype=f.dtype)
    dst = Array(name='dst', dimensions=dst_dimensions, dtype=f.dtype)

    if swap is False:
        eq = DummyEq(dst[dst_indices], src[src_indices])
        name = 'gather'
    else:
        eq = DummyEq(src[src_indices], dst[dst_indices])
        name = 'scatter'

    iet = Expression(eq)
    for d, dd in reversed(list(zip(f.dimensions, dst.dimensions))):
        iet = Iteration(iet, d.root, dd.symbolic_size)
    iet = List(body=[ArrayCast(src), ArrayCast(dst), iet])
    parameters = derive_parameters(iet)
    return Callable(name, iet, 'void', parameters, ('static',))


def mpi_irecv(buf, dest, comm):
    """
    Construct an IET performing an MPI_Irecv.
    """
    count = reduce(mul, buf.shape, 1)
    datatype = numpy_to_mpitypes(buf.dtype)
    args = [buf, count, datatype, DEST??, 'MPI_ANY_TAG', ]
    Call('MPI_Irecv', )


def mpi_exchange(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    assert f.is_Function

    # Construct send/recv buffers
    buffers = OrderedDict()
    for d0, side, region in product(f.dimensions, [LEFT, RIGHT], [OWNED, HALO]):
        if d0 in fixed:
            continue
        dimensions = []
        halo = []
        offsets = []
        for d1 in f.dimensions:
            if d1 in fixed:
                dimensions.append(DefaultDimension(name='h%s' % d1, default_value=1))
                halo.append((0, 0))
                offsets.append(fixed[d1])
            elif d0 is d1:
                offset, extent = f._get_region(region, d0, side, True)
                dimensions.append(DefaultDimension(name='h%s' % d1, default_value=extent))
                halo.append((0, 0))
                offsets.append(offset)
            else:
                dimensions.append(d1)
                halo.append(f._extent_halo[d0])
                offsets.append(0)
        array = Array(name='B%s%s' % (d0, side.name[0]), dimensions=dimensions,
                      halo=halo, dtype=f.dtype)
        buffers[(d0, side, region)] = (array, offsets)

    # Construct send/recv code
    groups = OrderedDict()
    for (d, side, region), (array, offsets) in buffers.items():
        # Call arguments
        args = [array] + list(array.symbolic_shape)
        args.extend([f] + offsets + list(f.symbolic_shape))

        # Function calls (gather/scatter and send/recv)
        if region is OWNED:
            call = Call('gather', args)
            
        else:
            call = Call('scatter', args)
        
        mask = Scalar(name='m%s%s' % (d, side.name[0]), dtype=np.int32)
        cond = Conditional(mask, call)
        from IPython import embed; embed()
