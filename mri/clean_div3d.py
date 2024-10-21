import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def clean_div3d(domain, coords, u_data):

    # unpack domain
    dist = domain.dist
    bases = domain.bases
    ybasis = bases[0]
    zbasis = bases[1]
    xbasis = bases[2]
    bases = (ybasis, zbasis, xbasis)

    ey = dist.VectorField(coords, name='ey')
    ez = dist.VectorField(coords, name='ez')
    ex = dist.VectorField(coords, name='ex')
    ey['g'][0] = 1
    ez['g'][1] = 1
    ex['g'][2] = 1

    y, z, x = dist.local_grids(ybasis, zbasis, xbasis)

    # Fields
    logger.info('divergence cleaning initiated...')
    ui = dist.VectorField(coords, name='ui', bases=bases)
    uc = dist.VectorField(coords, name='uc', bases=bases)
    ui['g'] = u_data.copy()
    pi = dist.Field(name='pi', bases=bases)
    tau_p = dist.Field(name='tau_p')
    tau_pi1 = dist.Field(name='tau_pi1', bases=(ybasis, zbasis))
    tau_pi2 = dist.Field(name='tau_pi1', bases=(ybasis, zbasis))

    lift_basis = xbasis.clone_with(a=1/2, b=1/2) # First derivative basis
    lift = lambda A, n: d3.Lift(A, lift_basis, n)
    grad_pi = d3.grad(pi) + ex*lift(tau_pi1, -1)

    # Problem
    problem = d3.LBVP([pi, uc, tau_pi1, tau_pi2, tau_p], namespace=locals())
    problem.add_equation("div(grad_pi) + lift(tau_pi2, -1) = -div(ui)")
    problem.add_equation("uc - grad(pi) = ui")

    problem.add_equation("integ(pi) = 0")
    problem.add_equation("(uc@ex)(x='left') = 0")
    problem.add_equation("(uc@ex)(x='right') = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve()
    logger.info('divergence cleaning successful')

    uc.change_scales(1)
    return uc['g'].copy()

# def clean_div3d_periodic(domain, coords, u_data):

#     # unpack domain
#     dist = domain.dist
#     bases = domain.bases
#     ybasis = bases[0]
#     zbasis = bases[1]
#     xbasis = bases[2]
#     bases = (ybasis, zbasis, xbasis)

#     y, z, x = dist.local_grids(ybasis, zbasis, xbasis)

#     # Fields
#     logger.info('divergence cleaning initiated...')
#     ui = dist.VectorField(coords, name='ui', bases=bases)
#     uc = dist.VectorField(coords, name='uc', bases=bases)
#     ui['g'] = u_data.copy()
#     pi = dist.Field(name='pi', bases=bases)
#     tau_p = dist.Field(name='tau_p')

#     # Problem
#     problem = d3.LBVP([pi, uc, tau_p], namespace=locals())
#     problem.add_equation("lap(pi) + tau_p = -div(ui)")
#     problem.add_equation("uc - grad(pi) = ui")

#     problem.add_equation("integ(pi) = 0")

#     # Solver
#     solver = problem.build_solver()
#     solver.solve()
#     logger.info('divergence cleaning successful')

#     uc.change_scales(1)
#     return uc['g'].copy()