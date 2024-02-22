import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys

def build_problem(domain, xcoord, a, b, c, *args):

    dealias = 3/2
    dtype = np.float64

    # unpack domain
    dist = domain.dist
    xbasis = domain.bases[0]
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Fields
    t = dist.Field()
    u_t = dist.Field(name='u_t', bases=xbasis)
    u = dist.Field(name='u', bases=xbasis)
    udiff = dist.Field(name='udiff', bases=xbasis)
    diffdiff = dist.Field(name='diffdiff', bases=xbasis)
    obj_t = dist.Field(name='obj_t', bases=xbasis)
    if (len(args) == 1):
        abber = args[0]
    else:
        abber = 0.0

    if (not isinstance(xbasis, d3.RealFourier)):
        tau_1 = dist.Field(name='tau_1')
        tau_2 = dist.Field(name='tau_2')     
        tau_3 = dist.Field(name='tau_3')

        lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        
        # Substitutions
        u_tx = dx(u_t) + lift(tau_1) # First-order reduction
        u_txx = dx(u_tx) + lift(tau_2) # First-order reduction

        # Problem
        problem = d3.IVP([u_t, tau_1, tau_2, tau_3], namespace=locals())
        problem.add_equation("dt(u_t) - a*u_txx + b * dx(u_txx) + lift(tau_3) = -c * u*dx(u_t)")

        # problem.add_equation("u_t(x='left') = 0")
        # problem.add_equation("u_t(x='right') = 0")
        # problem.add_equation("u_tx(x='right') = 0")

        problem.add_equation("u_t(x='left') - u_t(x='right') = 0")
        problem.add_equation("u_tx(x='left') - u_tx(x='right') = 0")
        problem.add_equation("u_txx(x='left') - u_txx(x='right') = 0")

        return problem

    # Problem
    # ft = np.heaviside(t - 5 + abber, 1.0)
    # ft = 5*np.exp((t-5)/abber)
    if abber == 0:
        ft = 0
    else:
        # ft = abber*t / 5.0
        # ft = 10*(np.tanh(30*(t - 5 + abber)) + 1) / 2.0
        ft = np.exp((t-5)/abber)
    problem = d3.IVP([u_t, obj_t], time=t, namespace=locals())


    # u - udiff = ubar
    if (np.isnan(abber)):
        problem.add_equation("dt(u_t) + a*dx(dx(u_t)) + b * dx(dx(dx(u_t))) = -c*u*dx(u_t) + udiff*dx(u - udiff)")
        problem.add_equation("dt(obj_t) = 0.0")
    elif (abber == 0):
        problem.add_equation("dt(u_t) + a*dx(dx(u_t)) + b * dx(dx(dx(u_t))) = -c*u*dx(u_t)")
        problem.add_equation("dt(obj_t) = 0.0")
    elif (abber < 0.0):
        problem.add_equation("dt(u_t) - a*(dx(dx(u_t - abber*dx(dx(u_t)) ))) + b * dx(dx(dx(u_t))) = -c*u*dx(u_t) - u_t*dx(u + u_t)")
        problem.add_equation("dt(obj_t) = 0.0")
    elif (abber == 1.0):
        problem.add_equation("dt(u_t) + a*dx(dx(u_t)) + b * dx(dx(dx(u_t))) = -c*dx(u*u_t) - abber*u_t*dx(u_t)")
        problem.add_equation("dt(obj_t) = 0.0")
    else:
        logger.info('linear simple backward integration!!!!')
        problem.add_equation("dt(u_t) + a*dx(dx(u_t)) + b * dx(dx(dx(u_t))) = -c*u*dx(u_t) - u_t*dx(u)")
        problem.add_equation("dt(obj_t) = 0.0")
    return problem