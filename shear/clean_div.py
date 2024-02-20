import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)


def clean_div(domain, coords, u_data):

    # unpack domain
    dist = domain.dist
    bases = domain.bases
    xbasis = bases[0]
    ybasis = bases[1]

    x, z = dist.local_grids(xbasis, ybasis)

    # Fields
    logger.info('divergence cleaning initiated...')
    ui = dist.VectorField(coords, name='ui', bases=(xbasis, ybasis))
    uc = dist.VectorField(coords, name='uc', bases=(xbasis, ybasis))
    ui['g'] = u_data.copy()
    pi = dist.Field(name='pi', bases=bases)
    tau_p = dist.Field(name='tau_p')

    # Problem
    problem = d3.LBVP([pi, uc, tau_p], namespace=locals())
    problem.add_equation("lap(pi) + tau_p = -div(ui)")
    problem.add_equation("uc - grad(pi) = ui")

    problem.add_equation("integ(pi) = 0")

    # Solver
    solver = problem.build_solver()
    solver.solve()
    logger.info('divergence cleaning successful')

    uc.change_scales(1)
    return uc['g'].copy()