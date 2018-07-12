from collections import OrderedDict
from functools import reduce
from itertools import product
from operator import mul
from ctypes import c_void_p

import numpy as np
from sympy import S

from devito.dimension import DefaultDimension, Dimension
from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import (ArrayCast, Call, Callable, Conditional,
                                 Expression, Iteration, List, PointerCast)
from devito.ir.iet.scheduler import iet_insert_C_decls
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import Byref, FieldFromPointer, Macro
from devito.types import Array, Symbol, LocalObject, OWNED, HALO
from devito.tools import is_integer, numpy_to_mpitypes

__all__ = ['copy', 'sendrecv', 'mpi_exchange']


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
        src_indices.append(d.root + Symbol(name='o%s' % d.root))
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
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable(name, iet, 'void', parameters, ('static',))


def sendrecv(f):
    """Construct an IET performing a halo exchange along arbitrary
    dimension and side."""
    assert f.is_Function
    assert f.grid is not None

    comm = f.grid.distributor._C_comm

    buf_dimensions = [Dimension(name='buf_%s' % d.root) for d in f.dimensions]
    bufg = Array(name='bufg', dimensions=buf_dimensions, dtype=f.dtype, scope='stack')
    bufs = Array(name='bufs', dimensions=buf_dimensions, dtype=f.dtype, scope='stack')

    dat_dimensions = [Dimension(name='dat_%s' % d.root) for d in f.dimensions]
    dat = Array(name='dat', dimensions=dat_dimensions, dtype=f.dtype,
                scope='external')

    ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
    ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

    params = [bufg] + list(bufg.symbolic_shape) + ofsg + [dat] + list(dat.symbolic_shape)
    gather = Call('gather', params)
    params = [bufs] + list(bufs.symbolic_shape) + ofss + [dat] + list(dat.symbolic_shape)
    scatter = Call('scatter', params)

    fromrank = Symbol(name='fromrank')
    torank = Symbol(name='torank')

    MPI_Request = type('MPI_Request', (c_void_p,), {})
    rrecv = LocalObject(name='rrecv', dtype=MPI_Request)
    rsend = LocalObject(name='rsend', dtype=MPI_Request)

    count = reduce(mul, bufs.symbolic_shape, 1)
    recv = Call('MPI_Irecv', [bufs, count, Macro(numpy_to_mpitypes(f.dtype)),
                              fromrank, Macro('MPI_ANY_TAG'), comm, rrecv])
    send = Call('MPI_Isend', [bufg, count, Macro(numpy_to_mpitypes(f.dtype)),
                              torank, Macro('MPI_ANY_TAG'), comm, rsend])

    waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
    waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

    iet = List(body=[recv, gather, send, waitrecv, waitsend, scatter])
    iet = iet_insert_C_decls(iet)
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable('sendrecv', iet, 'void', parameters, ('static',))


def mpi_exchange(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    # Requirements
    assert f.is_Function
    assert f.grid is not None

    distributor = f.grid.distributor
    nb = distributor._C_neighbours.obj
    comm = distributor._C_comm

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

    # If I receive on my right, then it means I'm also sending on my left
    mapper = {(LEFT, OWNED): LEFT, (RIGHT, OWNED): RIGHT,
              (LEFT, HALO): RIGHT, (RIGHT, HALO): LEFT}

    # Construct send/recv calls
    groups = OrderedDict()
    for (d, side, region), (array, offsets) in buffers.items():
        group = groups.setdefault((d, mapper[(side, region)]), {})

        # Pack-call arguments
        pack_args = [array] + list(array.symbolic_shape)
        pack_args.extend([f] + offsets + list(f.symbolic_shape))

        # MPI-call arguments
        count = reduce(mul, array.symbolic_shape, 1)
        dtype = Macro(numpy_to_mpitypes(array.dtype))
        peer = FieldFromPointer("%s%s" % (d, side.name), nb.name)
        mpi_args = [array, count, dtype, peer, Macro('MPI_ANY_TAG'), comm]

        # Function calls (gather/scatter and send/recv)
        if region is OWNED:
            rsend = Definition('MPI_Request', 'rsend')
            gather = Call('gather', pack_args)
            send = Call('MPI_Isend', mpi_args + [Byref('rsend')])
            wait = Call('MPI_Wait', [Byref('rsend'), Macro('MPI_STATUS_IGNORE')])
            group[OWNED] = [gather, rsend, send, wait]
        else:
            rrecv = Definition('MPI_Request', 'rrecv')
            scatter = Call('scatter', pack_args)
            recv = Call('MPI_Irecv', mpi_args + [Byref('rrecv')])
            wait = Call('MPI_Wait', [Byref('rrecv'), Macro('MPI_STATUS_IGNORE')])
            group[HALO] = [[rrecv, recv], [wait, scatter]]

    # Arrange send/recv into the most classical recv-send-wait pattern
    body = []
    for (d, side), blocks in groups.items():
        mask = Scalar(name='m%s%s' % (d, side.name[0]), dtype=np.int32)
        block = blocks[HALO][0] + blocks[OWNED] + blocks[HALO][1]
        body.append(Conditional(mask, block))

    # Build a Callable to invoke the newly-constructed halo exchange
    body = List(body=([PointerCast(comm), PointerCast(nb)] + body))
    parameters = derive_parameters(body, drop_locals=True)
    return Callable('halo_exchange', body, 'void', parameters, ('static',))
