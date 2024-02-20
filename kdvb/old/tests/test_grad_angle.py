import os
path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(path + "/..")
sys.path.append(path + "/../..")
from OptimizationContext import OptimizationContext
import numpy as np
import dedalus.public as d3
from dedalus.core.domain import Domain
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import Forward1D
import Backward1D
import matplotlib.pyplot as plt
from configparser import ConfigParser
import ast
import pytest

def test_solves1():
    filename = path + '/test_solves1_options.cfg'
    config = ConfigParser()
    config.read(str(filename))

    logger.info('Running diffusion_burgers.py with the following parameters:')
    logger.info(config.items('parameters'))

    write_suffix = str(config.get('parameters', 'suffix'))
    problem = str(config.get('parameters', 'problem'))
    if (len(sys.argv) >= 2):
        both = bool(int(sys.argv[1]))
    else:
        both = config.getboolean('parameters', 'both')

    write_objectives = config.getboolean('parameters', 'write_objectives')
    plot_paths = config.getboolean('parameters', 'plot_paths')
    show = config.getboolean('parameters', 'show')
    show_iter_cadence = config.getint('parameters', 'show_iter_cadence')
    show_loop_cadence = config.getint('parameters', 'show_loop_cadence')

    ks = ast.literal_eval(config.get('parameters', 'ks'))
    target_coeffs = ast.literal_eval(config.get('parameters', 'target_coeffs'))

    target = str(config.get('parameters', 'target'))
    alpha = config.getfloat('parameters', 'alpha')
    beta = config.getfloat('parameters', 'beta')


    if (len(sys.argv) >= 3):
        abber = float(sys.argv[2])
    else:
        abber = config.getfloat('parameters', 'abber')

    # Parameters
    Lx = eval(config.get('parameters', 'Lx'))
    N = config.getint('parameters', 'Nx')

    a = config.getfloat('parameters', 'a')
    b = config.getfloat('parameters', 'b')
    c = config.getfloat('parameters', 'c')
    T = eval(config.get('parameters', 'T'))
    dt = eval(config.get('parameters', 'dt'))

    opt_iters = config.getint('parameters', 'opt_iters')
    num_cp = config.getint('parameters', 'num_cp')

    alpha_str = str(alpha).replace('.', 'p')
    abber_str = str(abber).replace('.', 'p')

    # Simulation Parameters
    dealias = 3/2
    dtype = np.float64

    periodic = config.getboolean('parameters', 'periodic')
    gamma_init = config.getfloat('parameters', 'gamma_init')
    epsilon_safety = default_gamma = 0.6

    # Bases
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64, comm=MPI.COMM_SELF)

    if (periodic):
        xbasis = d3.RealFourier(xcoord, size=N, bounds=(0, Lx), dealias=3/2)
    else:
        xbasis = d3.ChebyshevT(xcoord, size=N, bounds=(0, Lx), dealias=3/2)

    domain = Domain(dist, [xbasis])
    dist = domain.dist

    x = dist.local_grid(xbasis)

    logger.info('Building problem {} with basis {}'.format(problem, type(xbasis).__name__))
    if problem == 'kdv':
        forward_problem = Forward1D.build_problem(domain, xcoord, a, b, c, alpha)
        logger.info('abberation param: {}'.format(abber))
        backward_problem = Backward1D.build_problem(domain, xcoord, a, b, c, abber)

    else:
        logger.error('problem not recognized')
        raise

    lagrangian_dict = {forward_problem.variables[0] : backward_problem.variables[0]}

    forward_solver = forward_problem.build_solver(d3.RK443)
    # udiff = forward_solver.state[1]
    backward_solver = backward_problem.build_solver(d3.RK443)

    if (show):
        from Animate1D import Animate1D
        opt = Animate1D(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
    else:
        opt = OptimizationContext(domain, xcoord, forward_solver, backward_solver, lagrangian_dict, None, write_suffix)
    opt.set_time_domain(T, num_cp, dt)
    opt.x_grid = x

    opt.show_iter_cadence = show_iter_cadence
    opt.show_loop_cadence = show_loop_cadence
    opt.show = show

    logger.info('using target soln: {}'.format(target))
    if (target == 'analytic'):
        U0g = beta + (alpha - beta) * (np.cosh(np.sqrt((alpha - beta)/(12*b)) * (x - np.pi)))**(-2.0)
    else:
        U0g = np.zeros_like(x)
        for i in range(len(target_coeffs)):
            U0g += target_coeffs[i] * np.sin(ks[i] * 2*np.pi * x / Lx)

    # opt.ic['u']['g'] = U0g.copy()
    ubar0 = dist.Field(name='ubar0', bases=xbasis)
    ubar0['g'] = U0g.copy()

    opt.soln = U0g.copy()
    u = next(field for field in forward_solver.state if field.name == 'u')
    objectiveT = 0.5*(u - ubar0)**2
    opt.set_objectiveT(objectiveT)

    # f = opt.loop(U0g.copy())
    # fprime = opt.jac(opt.x)

    # logger.info(f)
    # logger.info(np.max(fprime**2))

    # Failed = np.abs(f) < 1e-7
    # Failed *= np.max(fprime**2) < 1e-7

    mode1 = dist.Field(name='mode1', bases=xbasis)
    mode1['g'] = 0.1*np.sin(x)
    f = opt.loop(U0g.copy() + mode1['g'].copy())
    fprime = opt.jac(opt.x)

    logger.info(f)
    logger.info(np.max(fprime**2))
    logger.info(d3.Integrate((backward_problem.variables[0] + mode1)**2).evaluate()['g'].flat[0])
    # logger.info(d3.Integrate(backward_problem.variables[0]*mode1).evaluate()['g'].flat[0])
    Failed = False
    # Failed *= (f - 0.006127716521439899) < 1e-7
    # Failed *= (np.max(fprime**2) - 0.00792293211706787) < 1e-7

    # f = opt.loop(np.zeros_like(U0g))
    # fprime = opt.jac(opt.x)

    # logger.info(f)
    # logger.info(np.max(fprime**2))

    # Failed *= f == 0
    # Failed *= np.max(fprime**2) == 0
    logger.info('test passed = {}'.format(Failed))

if __name__ == "__main__":
    test_solves1()