import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt
from dedalus.core.operators import TimeDerivative

def build_problem(domain, coords, params):

    # unpack domain
    dist = domain.dist
    bases = domain.bases
    ybasis, zbasis, xbasis = bases[0], bases[1], bases[2]
    y, z, x = dist.local_grids(ybasis, zbasis, xbasis)
    ey = dist.VectorField(coords, name='ey')
    ez = dist.VectorField(coords, name='ez')
    ex = dist.VectorField(coords, name='ex')
    ey['g'][0] = 1
    ez['g'][1] = 1
    ex['g'][2] = 1

    # nccs
    U0 = dist.VectorField(coords, name='U0', bases=xbasis)
    U0['g'][0] = params["S"] * x

    # B0 = 0
    # B0 = dist.VectorField(coords, name='B0', bases=xbasis)
    # B0['g'][1] = 0

    fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
    fz_hat['g'][1] = params["f"]

    # damping timescale
    TAU = dist.Field(name='TAU')

    # Fields
    p = dist.Field(name='p', bases=bases)
    phi = dist.Field(name='phi', bases=bases)
    u = dist.VectorField(coords, name='u', bases=bases)
    A = dist.VectorField(coords, name='A', bases=bases)
    taup = dist.Field(name='taup')
    tau1u = dist.VectorField(coords, name='tau1u', bases=(ybasis,zbasis))
    tau2u = dist.VectorField(coords, name='tau2u', bases=(ybasis,zbasis))
    tau1A = dist.VectorField(coords, name='tau1A', bases=(ybasis,zbasis))
    tau2A = dist.VectorField(coords, name='tau2A', bases=(ybasis,zbasis))

    lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)


    # operations
    # b.store_last = True
    # b = d3.Curl(A) + ex*lift(tau1A,-1)
    b = d3.Curl(A)
    integy = lambda A: d3.Integrate(A, 'y')
    integz = lambda A: d3.Integrate(A, 'z')
    integx = lambda A: d3.Integrate(A, 'x')
    integ = lambda A: integy(integz(integx(A)))

    dx = lambda A: d3.Differentiate(A, coords['x'])
    dt = lambda argy: TimeDerivative(argy)
    grad_u = d3.grad(u) + ex*lift(tau1u,-1) # First-order reduction
    grad_A = d3.grad(A) + ex*lift(tau1A,-1) # First-order reduction
    grad_b = d3.grad(b)
    # grad_B0 = d3.grad(B0)

    Ly  = params["Ly"]
    Lz  = params["Lz"]
    Lx  = params["Lx"]
    nu  = params["nu"]
    eta = params["eta"]
    tau = params["tau"]


    DIVU_LHS = d3.trace(grad_u) + taup
    DIVU_RHS = 0

    DIVA_LHS = d3.trace(grad_A)
    DIVA_RHS = 0

    NS_LHS = dt(u) - nu*d3.div(grad_u) + d3.grad(p) + d3.cross(fz_hat, u) + lift(tau2u,-1) 
    NS_RHS = d3.cross(u, d3.curl(u)) - d3.cross(b, d3.curl(b))
    # NS_RHS = d3.dot(b, grad_b) - d3.dot(u, grad_u)

    # NS_LHS += d3.cross(B0, d3.curl(d3.curl(A))) + d3.cross(d3.curl(A), d3.curl(B0))
    NS_LHS += d3.dot(u, d3.grad(U0)) + d3.dot(U0, grad_u)

    if tau > 0:
        TAU['g'] = tau
        NS_LHS += u / TAU

    elif tau < 0:
        # using the sign of tau here to trigger averaging
        TAU['g'] = -tau
        NS_LHS += integy(integz(u)) / TAU / Ly / Lz

    IND_LHS = dt(A) + d3.grad(phi) - eta*d3.div(grad_A) + lift(tau2A,-1) - d3.cross(U0, d3.curl(A))
    IND_RHS = d3.cross(u, b)

    problem = d3.IVP([p, phi, u, A, taup, tau1u, tau2u, tau1A, tau2A], namespace=locals())
    problem.add_equation((DIVU_LHS, DIVU_RHS))
    problem.add_equation((DIVA_LHS, DIVA_RHS))
    problem.add_equation((NS_LHS,   NS_RHS))
    problem.add_equation((IND_LHS,  IND_RHS))

    # b.grad(b) = j x b
    # u.grad(u) = omega x u

    # problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) - nu*div(grad_u) + grad(p) + cross(fz_hat, u) + lift(tau2u,-1) - dot(curl(A), grad_B0) = dot(B0, grad_b) + dot(b, grad_b) - dot(u,grad(u))")
    # problem.add_equation("dt(A) + grad(phi) - eta*div(grad_A) + lift(tau2A,-1) - cross(U0, curl(A)) - cross(u, B0) = cross(u, b)")
    # # 30 transforms per timestep

    # problem.add_equation("dt(u) + dot(u,grad(U0)) + dot(U0,grad(u)) - nu*div(grad_u) + grad(p) + cross(fz_hat, u) + lift(tau2u,-1) - dot(curl(A), grad_B0) = dot(B0, grad_b) + dot(b, grad_b) - dot(u,grad(u))")
    # problem.add_equation("dt(A) + grad(phi) - eta*div(grad_A) + lift(tau2A,-1) - cross(U0, curl(A)) - cross(u, B0) = cross(u, b)")
    # # 30 transforms per timestep

    if (params["isNoSlip"]):
        # no-slip BCs
        problem.add_equation("u(x='left')  = 0")
        problem.add_equation("u(x='right') = 0")
    else:
        # stress-free BCs
        if True:
            problem.add_equation("dot(u, ex)(x='left')      = 0")
            problem.add_equation("dot(u, ex)(x='right')     = 0")
            problem.add_equation("dot(dx(u), ey)(x='left')  = 0")
            problem.add_equation("dot(dx(u), ey)(x='right') = 0")
            problem.add_equation("dot(dx(u), ez)(x='left')  = 0")
            problem.add_equation("dot(dx(u), ez)(x='right') = 0")
        else:
            problem.add_equation("dot(u, ex)(x='left')      = dot(U0, ex)(x='left')")
            problem.add_equation("dot(u, ex)(x='right')     = dot(U0, ex)(x='right')")
            problem.add_equation("dot(dx(u), ey)(x='left')  = dot(dx(U0), ey)(x='left')")
            problem.add_equation("dot(dx(u), ey)(x='right') = dot(dx(U0), ey)(x='right')")
            problem.add_equation("dot(dx(u), ez)(x='left')  = dot(dx(U0), ez)(x='left')")
            problem.add_equation("dot(dx(u), ez)(x='right') = dot(dx(U0), ez)(x='right')")

    problem.add_equation("integ(p)       = 0") 
    problem.add_equation("phi(x='left')  = 0")
    problem.add_equation("phi(x='right') = 0")

    problem.add_equation("dot(A, ey)(x='left')  = 0")
    problem.add_equation("dot(A, ez)(x='left')  = 0")
    problem.add_equation("dot(A, ey)(x='right') = 0")
    problem.add_equation("dot(A, ez)(x='right') = 0")

    return problem