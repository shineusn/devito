from functools import reduce
from operator import mul
from ctypes import c_void_p

from sympy import Ge

from devito.dimension import Dimension
from devito.mpi.utils import get_views
from devito.ir.equations import DummyEq
from devito.ir.iet import (ArrayCast, Call, Callable, Conditional, Expression,
                           Iteration, List, iet_insert_C_decls,
                           derive_parameters)
from devito.symbolics import CondNe, FieldFromPointer, FieldFromComposite, Macro
from devito.types import Array, Symbol, LocalObject, OWNED, HALO, LEFT, RIGHT
from devito.tools import numpy_to_mpitypes


from devito.ir.iet import Element
import cgen as c


__all__ = ['copy', 'sendrecv', 'update_halo']


def copy(f, fixed, swap=False):
    """
    Construct a :class:`Callable` capable of copying: ::

        * an arbitrary convex region of ``f`` into a contiguous :class:`Array`, OR
        * if ``swap=True``, a contiguous :class:`Array` into an arbitrary convex
          region of ``f``.
    """
    buf_dims = []
    buf_indices = []
    for d in f.dimensions:
        if d not in fixed:
            buf_dims.append(Dimension(name='buf_%s' % d.root))
            buf_indices.append(d.root)
    buf = Array(name='buf', dimensions=buf_dims, dtype=f.dtype)

    dat_dims = []
    dat_offsets = []
    dat_indices = []
    for d in f.dimensions:
        dat_dims.append(Dimension(name='dat_%s' % d.root))
        offset = Symbol(name='o%s' % d.root)
        dat_offsets.append(offset)
        dat_indices.append(offset + (d.root if d not in fixed else 0))
    dat = Array(name='dat', dimensions=dat_dims, dtype=f.dtype)

    if swap is False:
        eq = DummyEq(buf[buf_indices], dat[dat_indices])
        name = 'gather_%s' % f.name
        a = Element(c.Statement('printf("gather! ")'))
    else:
        eq = DummyEq(dat[dat_indices], buf[buf_indices])
        name = 'scatter_%s' % f.name
        a = Element(c.Statement('printf("scatter! ")'))


    a2 = Element(c.Statement('printf("buf_x_size=%d ", buf_x_size)'))
    a3 = Element(c.Statement('printf("dat_x_size=%d ", dat_x_size)'))
    a4 = Element(c.Statement('printf("otime=%d ", otime)'))
    a5 = Element(c.Statement('printf("ox=%d", ox)'))
    a6 = Element(c.Statement('printf("buf[0]=%e\\n", buf[0])'))

    iet = Expression(eq)
    for i, d in reversed(list(zip(buf_indices, buf_dims))):
        iet = Iteration(iet, i, d.symbolic_size - 1)  # -1 as Iteration generates <=
    iet = List(body=[ArrayCast(dat), ArrayCast(buf), iet])
    #iet = List(body=[ArrayCast(dat), ArrayCast(buf), a, a2, a3, a4, a5, iet, a6])
    parameters = [buf] + list(buf.shape) + [dat] + list(dat.shape) + dat_offsets
    return Callable(name, iet, 'void', parameters, ('static',))


def sendrecv(f, fixed):
    """Construct an IET performing a halo exchange along arbitrary
    dimension and side."""
    assert f.is_Function
    assert f.grid is not None

    comm = f.grid.distributor._C_comm

    buf_dims = [Dimension(name='buf_%s' % d.root) for d in f.dimensions if d not in fixed]
    bufg = Array(name='bufg', dimensions=buf_dims, dtype=f.dtype, scope='stack')
    bufs = Array(name='bufs', dimensions=buf_dims, dtype=f.dtype, scope='stack')

    dat_dims = [Dimension(name='dat_%s' % d.root) for d in f.dimensions]
    dat = Array(name='dat', dimensions=dat_dims, dtype=f.dtype, scope='external')

    ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
    ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

    fromrank = Symbol(name='fromrank')
    torank = Symbol(name='torank')

    parameters = [bufg] + list(bufg.shape) + [dat] + list(dat.shape) + ofsg
    gather = Call('gather_%s' % f.name, parameters)
    parameters = [bufs] + list(bufs.shape) + [dat] + list(dat.shape) + ofss
    scatter = Call('scatter_%s' % f.name, parameters)

    # The scatter must be guarded as we must not alter the halo values along
    # the domain boundary, where the sender is actually MPI.PROC_NULL
    scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)
#    from devito.ir.iet import Element
#    import cgen as c
#    a2 = Element(c.Statement('printf("MPI_PROC_NUL=%d\\n", MPI_PROC_NULL)'))
    a = Element(c.Statement('printf("bufs[0]=%e ", bufs[0])'))
    a1 = Element(c.Statement('printf("fromrank=%d ", fromrank)'))
    a2 = Element(c.Statement('printf("torank=%d ", torank)'))
    scatter = List(body=[scatter])

    MPI_Status = type('MPI_Status', (c_void_p,), {})
    srecv = LocalObject(name='srecv', dtype=MPI_Status)
    import numpy as np
    count = LocalObject(name='count', dtype=np.int)
    from devito.symbolics import Byref
    bb = Call('MPI_Get_count', [srecv, Macro(numpy_to_mpitypes(f.dtype)), count])
    #a3 = Element(c.Statement('printf("count=%d \\n", count)'))
    bb = Element(c.Statement('int count'))
    a3 = Element(c.Statement('MPI_Comm_size(comm, &count)'))
    a33 = Element(c.Statement('printf("count=%d \\n", count)'))
    a4 = Conditional(CondNe(FieldFromComposite('MPI_ERROR', srecv), Macro('MPI_SUCCESS')),
                     List(body=[Element(c.Statement('printf("WTF!?!?\\n")'))]))
    cc = List(body=[bb, a1, a2, a, a3, a33])
    dd = Element(c.Statement('printf("Waitrecv done !!!\\n")'))

    MPI_Request = type('MPI_Request', (c_void_p,), {})
    rrecv = LocalObject(name='rrecv', dtype=MPI_Request)
    rsend = LocalObject(name='rsend', dtype=MPI_Request)

    count = reduce(mul, bufs.shape, 1)
    recv = Call('MPI_Irecv', [bufs, count, Macro(numpy_to_mpitypes(f.dtype)),
#                              fromrank, Macro('MPI_ANY_TAG'), comm, rrecv])
                              fromrank, '13', comm, rrecv])
    send = Call('MPI_Isend', [bufg, count, Macro(numpy_to_mpitypes(f.dtype)),
#                              torank, Macro('MPI_ANY_TAG'), comm, rsend])
                              torank, '13', comm, rsend])

    waitrecv = Call('MPI_Wait', [rrecv, srecv])
    waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

    iet = List(body=[recv, gather, send, waitsend, cc, waitrecv, dd, scatter])
    iet = List(body=[ArrayCast(dat), iet_insert_C_decls(iet)])
    parameters = ([dat] + list(dat.shape) + list(bufs.shape) +
                  ofsg + ofss + [fromrank, torank, comm])
    return Callable('sendrecv_%s' % f.name, iet, 'void', parameters, ('static',))


def update_halo(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    # Requirements
    assert f.is_Function
    assert f.grid is not None

    distributor = f.grid.distributor
    nb = distributor._C_neighbours.obj
    comm = distributor._C_comm

    fixed = {d: Symbol(name="o%s" % d.root) for d in fixed}

    mapper = get_views(f, fixed)

    body = []
    for d in f.dimensions:
        if d in fixed:
            continue

        rpeer = FieldFromPointer("%sright" % d, nb)
        lpeer = FieldFromPointer("%sleft" % d, nb)

        # Sending to left, receiving from right
        lsizes, loffsets = mapper[(d, LEFT, OWNED)]
        rsizes, roffsets = mapper[(d, RIGHT, HALO)]
        assert lsizes == rsizes
        sizes = lsizes
        parameters = ([f] + list(f.symbolic_shape) + sizes + loffsets +
                      roffsets + [rpeer, lpeer, comm])
        call = Call('sendrecv_%s' % f.name, parameters)
        body.append(Conditional(Symbol(name='m%sl' % d), call))

        # Sending to right, receiving from left
        rsizes, roffsets = mapper[(d, RIGHT, OWNED)]
        lsizes, loffsets = mapper[(d, LEFT, HALO)]
        assert rsizes == lsizes
        sizes = rsizes
        parameters = ([f] + list(f.symbolic_shape) + sizes + roffsets +
                      loffsets + [lpeer, rpeer, comm])
        call = Call('sendrecv_%s' % f.name, parameters)
        body.append(Conditional(Symbol(name='m%sr' % d), call))

    iet = List(body=body)
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable('halo_exchange_%s' % f.name, iet, 'void', parameters, ('static',))
