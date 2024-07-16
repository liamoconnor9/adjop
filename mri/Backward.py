import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt

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
    U0['g'][0] = params["S"] * x * 0

    # B0 = 0
    # B0 = dist.VectorField(coords, name='B0', bases=xbasis)
    # B0['g'][1] = 0

    fz_hat = dist.VectorField(coords, name='fz_hat', bases=xbasis)
    fz_hat['g'][1] = params["f"]

    # damping timescale
    TAU = dist.Field(name='TAU')

    # Adjoint Fields
    p_t = dist.Field(name='p_t', bases=bases)
    phi_t = dist.Field(name='phi_t', bases=bases)
    u_t = dist.VectorField(coords, name='u_t', bases=bases)
    A_t = dist.VectorField(coords, name='A_t', bases=bases)
    b_t = dist.VectorField(coords, name='b_t', bases=bases)
    taup_t = dist.Field(name='taup_t')
    tau1u_t = dist.VectorField(coords, name='tau1u_t', bases=(ybasis,zbasis))
    tau2u_t = dist.VectorField(coords, name='tau2u_t', bases=(ybasis,zbasis))
    tau1A_t = dist.VectorField(coords, name='tau1A_t', bases=(ybasis,zbasis))
    tau2A_t = dist.VectorField(coords, name='tau2A_t', bases=(ybasis,zbasis))

    # Primative Fields
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
    # b_t = d3.Curl(A_t)
    b = d3.Curl(A)
    integy = lambda A: d3.Integrate(A, 'y')
    integz = lambda A: d3.Integrate(A, 'z')
    integx = lambda A: d3.Integrate(A, 'x')
    integ = lambda A: integy(integz(integx(A)))

    dx = lambda A: d3.Differentiate(A, coords['x'])
    dt = lambda argy: TimeDerivative(argy)
    grad_u_t = d3.grad(u_t) + ex*lift(tau1u_t,-1) # First-order reduction
    grad_A_t = d3.grad(A_t) + ex*lift(tau1A_t,-1) # First-order reduction
    grad_b_t = d3.grad(b_t)
    # grad_B0 = d3.grad(B0)

    Ly  = params["Ly"]
    Lz  = params["Lz"]
    Lx  = params["Lx"]
    nu  = params["nu"]
    eta = params["eta"]
    tau = params["tau"]

    N = lambda YY_t, YY: d3.cross(d3.curl(YY), YY_t) + d3.curl(d3.cross(YY_t, YY))

    DIVU_LHS = d3.trace(grad_u_t) + taup_t
    DIVU_RHS = 0

    DIVA_LHS = d3.trace(grad_A_t)
    DIVA_RHS = 0

    NS_LHS = dt(u_t) + d3.grad(p_t) + d3.cross(fz_hat, u_t) + nu*d3.div(grad_u_t) + N(u_t, U0) + lift(tau2u_t,-1) 
    NS_RHS = -N(u_t, u) - d3.cross(b, d3.curl(b_t))

    if tau > 0:
        TAU['g'] = tau
        NS_LHS -= u_t / TAU

    # elif tau < 0:
    #     # using the sign of tau here to trigger averaging
    #     TAU['g'] = -tau
    #     NS_LHS -= integy(integz(u_t)) / TAU / Ly / Lz

    # IND_LHS = dt(A_t) + d3.grad(phid) - eta*d3.div(grad_A_t) + lift(tau2A,-1) - d3.cross(U0, d3.curl(A))
    # IND_RHS = d3.cross(u, b)

# 0 = -\nabla\tilde{\phi} + \mathbf{\tilde{A}}\times(\nabla\times\mathbf{A})-\partial_t\mathbf{\tilde{A}} -\nabla\times(\mathbf{\tilde{A}}\times \mathbf{u}) - \nabla\times\left( \mathbf{\tilde{u}}\times(\nabla\times\nabla\times\mathbf{A})\right) - \nabla^2\left( (\nabla\times\mathbf{A}) \times \mathbf{\tilde{u}}  \right) - \nu\nabla^2\mathbf{\tilde{A}}

    IND_LHS = -dt(A_t) - d3.grad(phi_t) - nu*d3.div(grad_A_t) + lift(tau2A_t,-1) 
    IND_RHS = d3.curl(d3.cross(A_t, u)) + d3.curl(d3.cross(u_t, d3.curl(d3.curl(A)))) + d3.div(d3.grad(d3.cross(d3.curl(A), u_t)))
    problem = d3.IVP([p_t, phi_t, u_t, A_t, taup_t, tau1u_t, tau2u_t, tau1A_t, tau2A_t], namespace=locals())
    problem.add_equation((DIVU_LHS, DIVU_RHS))
    problem.add_equation((DIVA_LHS, DIVA_RHS))
    problem.add_equation((NS_LHS,   NS_RHS))
    problem.add_equation((IND_LHS,  IND_RHS))

    if (params["isNoSlip"]):
        # no-slip BCs
        problem.add_equation("u_t(x='left')  = 0")
        problem.add_equation("u_t(x='right') = 0")
    else:
        # stress-free BCs
        if True:
            problem.add_equation("dot(u_t, ex)(x='left')      = 0")
            problem.add_equation("dot(u_t, ex)(x='right')     = 0")
            problem.add_equation("dot(dx(u_t), ey)(x='left')  = 0")
            problem.add_equation("dot(dx(u_t), ey)(x='right') = 0")
            problem.add_equation("dot(dx(u_t), ez)(x='left')  = 0")
            problem.add_equation("dot(dx(u_t), ez)(x='right') = 0")
        else:
            problem.add_equation("dot(u_t, ex)(x='left')      = dot(u_t0, ex)(x='left')")
            problem.add_equation("dot(u_t, ex)(x='right')     = dot(u_t0, ex)(x='right')")
            problem.add_equation("dot(dx(u_t), ey)(x='left')  = dot(dx(u_t0), ey)(x='left')")
            problem.add_equation("dot(dx(u_t), ey)(x='right') = dot(dx(u_t0), ey)(x='right')")
            problem.add_equation("dot(dx(u_t), ez)(x='left')  = dot(dx(u_t0), ez)(x='left')")
            problem.add_equation("dot(dx(u_t), ez)(x='right') = dot(dx(u_t0), ez)(x='right')")

    problem.add_equation("integ(p_t)       = 0") 
    problem.add_equation("phi_t(x='left')  = 0")
    problem.add_equation("phi_t(x='right') = 0")
    
    problem.add_equation("dot(A_t, ey)(x='left')  = 0")
    problem.add_equation("dot(A_t, ez)(x='left')  = 0")
    problem.add_equation("dot(A_t, ey)(x='right') = 0")
    problem.add_equation("dot(A_t, ez)(x='right') = 0")

    # problem.add_equation("dot(b_t, ex)(x='left')  = 0")
    # problem.add_equation("dot(b_t, ex)(x='right') = 0")
    # problem.add_equation("dot(dx(b_t), ex)(x='left')  = 0")
    # problem.add_equation("dot(dx(b_t), ex)(x='right') = 0")

    # problem.add_equation("phi_t(x='left')  = 0")
    # problem.add_equation("phi_t(x='right') = 0")

    return problem
















def build_shear_problem(domain, coords, Reynolds, *args):

    if (len(args) == 1):
        abber = args[0]
    else:
        abber = 0

    # unpack domain
    dist = domain.dist
    bases = domain.bases
        
    # Fields
    p_t = dist.Field(name='p_t', bases=bases)
    s_t = dist.Field(name='s_t', bases=bases)
    obj_t = dist.Field(name='obj_t', bases=bases)
    u_t = dist.VectorField(coords, name='u_t', bases=bases)

    u = dist.VectorField(coords, name='u', bases=bases)
    s = dist.Field(name='s', bases=bases)

    tau_p_t = dist.Field(name='tau_p_t')

    # Substitutions
    nu = 1 / Reynolds
    Schmidt = 1
    D = nu / Schmidt

    # Problem
    if (abber == 0):
        logger.info('abber = 0')
        problem = d3.IVP([u_t, s_t, p_t, tau_p_t], namespace=locals())
        problem.add_equation("dt(u_t) + grad(p_t) + nu*lap(u_t) = u_t@transpose(grad(u)) - u@grad(u_t)")
        problem.add_equation("dt(s_t) + D*lap(s_t) = -u@grad(s_t)")
        problem.add_equation("div(u_t) + tau_p_t = 0")
        problem.add_equation("integ(p_t) = 0") # Pressure gauge
        return problem
    
    # SBI without nonlinearity
    elif (abber == -1):
        logger.info('abber = {}'.format(abber))
        logger.info('running linear SBI')
        problem = d3.IVP([u_t, s_t, p_t, obj_t, tau_p_t], namespace=locals())
        problem.add_equation("dt(u_t) + grad(p_t) + nu*lap(u_t) = -u_t@(grad(u)) - u@grad(u_t)")
        problem.add_equation("dt(s_t) + D*lap(s_t) = -u@grad(s_t)")
        problem.add_equation("div(u_t) + tau_p_t = 0")
        problem.add_equation("integ(p_t) = 0") # Pressure gauge

    # QRM with vareps as small param
    elif (abber < 0):
        vareps = -abber
        logger.info('abber = {}'.format(abber))
        logger.info('using quasireversibility method with var epsilon = {}'.format(vareps))
        problem = d3.IVP([u_t, s_t, p_t, obj_t, tau_p_t], namespace=locals())
        problem.add_equation("dt(s_t) + D*lap(s_t) = -u@grad(s_t)")
        problem.add_equation("div(u_t) + tau_p_t = 0")
        problem.add_equation("integ(p_t) = 0") # Pressure gauge

        # old adj
        problem.add_equation("dt(u_t) + grad(p_t) - nu*lap(u_t + vareps*lap(u_t)) = -u_t@(grad(u)) - u@grad(u_t) - ( u_t@grad(u_t))")
        problem.add_equation("dt(obj_t) = -abber*( ((u + u_t)@grad(u_t))@u_t + (u_t@grad(u + u_t))@u_t + nu * u_t @ lap(u_t))")
        return problem
    else:
        logger.info('abber = {}'.format(abber))
        problem = d3.IVP([u_t, s_t, p_t, obj_t, tau_p_t], namespace=locals())
        # problem.add_equation("dt(u_t) + grad(p_t) + nu*lap(u_t) = u_t@transpose(grad(u)) - u@grad(u_t) - abber*( u_t@grad(u + u_t))")
        problem.add_equation("dt(s_t) + D*lap(s_t) = -u@grad(s_t)")
        problem.add_equation("div(u_t) + tau_p_t = 0")
        problem.add_equation("integ(p_t) = 0") # Pressure gauge
        # problem.add_equation("dt(obj_t) = -abber*( u'i*(uj - u'j)*dj(u'i) + u'i*u'j*dj(ui - u'i) + nu * u_t @ lap(u_t))")
        # problem.add_equation("dt(obj_t) = -abber*( u_ti*(uj + u_tj)*dj(u_ti) + u_ti*u_tj*dj(ui + u_ti) + nu * u_t @ lap(u_t))")
        # problem.add_equation("dt(obj_t) = -abber*(nu * u_t @ lap(u_t))")
        # problem.add_equation("dt(obj_t) = -abber*( ((u + u_t)@grad(u_t))@u_t + (u_t*grad(u + u_t))@u_t + nu * u_t @ lap(u_t))")

        problem.add_equation("dt(u_t) + grad(p_t) + nu*lap(u_t) = -u_t@(grad(u)) - u@grad(u_t) - abber*( u_t@grad(u_t))")
        problem.add_equation("dt(obj_t) = -abber*( ((u + u_t)@grad(u_t))@u_t + (u_t@grad(u + u_t))@u_t + nu * u_t @ lap(u_t))")
        return problem
