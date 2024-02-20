import numpy as np
import dedalus.public as d3
from mpi4py import MPI
CW = MPI.COMM_WORLD
import logging
logger = logging.getLogger(__name__)
import sys
import matplotlib.pyplot as plt

def build_problem(domain, coords, Reynolds, *args):

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
