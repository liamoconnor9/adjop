import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt

def build_problem(domain, xcoord, a, b, c, *args):

    # unpack domain
    dist = domain.dist
    xbasis = domain.bases[0]
    Lx = 2*np.pi
    
    u = dist.Field(name='u', bases=xbasis)
    U = dist.Field(name='U', bases=xbasis)
    t = dist.Field()
    xf = dist.Field(name='xf', bases=xbasis)
    xf['g'] = xbasis.local_grid().copy()
    # print(np.min(xf['g']))
    # sys.exit()
    # x = dist.local_grid(xbasis)


    if (len(args) == 1):
        alpha = args[0]
    else:
        alpha = 1
    logger.info('alpha = {}'.format(alpha))
    beta = 0.0
    arg = (xf - alpha*0/3.0 - np.pi)
    # U = (beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(-12*b)) * (arg)))**(-2.0))
    # for i in range(1,3):
    #     U += (beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(-12*b)) * (arg + i*Lx)))**(-2.0))

    udiff = dist.Field(name='udiff', bases=xbasis)
    dx = lambda A: d3.Differentiate(A, xcoord)

    if (not isinstance(xbasis, d3.RealFourier)):
        tau_1 = dist.Field(name='tau_1')
        tau_2 = dist.Field(name='tau_2')     
        tau_3 = dist.Field(name='tau_3')

        lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
        lift = lambda A: d3.Lift(A, lift_basis, -1)
        
        # Substitutions
        ux = dx(u) + lift(tau_1) # First-order reduction
        uxx = dx(ux) + lift(tau_2) # First-order reduction

        # Problem
        problem = d3.IVP([u, udiff, tau_1, tau_2, tau_3], time=t, namespace=locals())
        problem.add_equation("dt(u) - a*uxx + b*dx(uxx) + lift(tau_3) = -c*u*ux")
        problem.add_equation("udiff = u - U")


        # problem.add_equation("u(x='left') = 0")
        # problem.add_equation("u(x='right') = 0")
        # problem.add_equation("ux(x='left') = 0")

        problem.add_equation("u(x='left') - u(x='right') = 0")
        problem.add_equation("ux(x='left') - ux(x='right') = 0")
        problem.add_equation("uxx(x='left') - uxx(x='right') = 0")
        # problem.add_equation("uxx(x='left') - uxx(x='right') = 0")
        return problem
        

    # Problem

    problem = d3.IVP([u], time=t, namespace=locals())
    problem.add_equation("dt(u) - a*dx(dx(u)) + b*dx(dx(dx(u))) = -c*u*dx(u)")
    # problem.add_equation("dt(U) + alpha/3.0*dx(U) = 0")
    # problem.add_equation("udiff = u - U")
    return problem