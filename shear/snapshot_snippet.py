
# This code snippet is evaluated around line 130 of shear_flow.py 
# when parameter "snapshot_snippet" is set to this file's name.

# (shear_flow.py runs the forward simulation to generate the target end-state & snapshots at a given loop index)

logger.info("EVALUATING SNAPSHOT SNIPPET IN FILE: {}".format(snapshots_snippet))

if ('_prod' in suffix):
    snapshot_scales = round(1024 / Nx)
    logger.info('writing snapshots for production (1024 x 2048) by setting scales = {}'.format(snapshot_scales))
    snapshots = solver.evaluator.add_file_handler(path + '/' + suffix + '/snapshots_{}'.format(label), sim_dt=0.01, max_writes=5)
    snapshots.add_task(s, name='tracer', layout='g', scales=snapshot_scales)
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity', layout='g', scales=snapshot_scales)
else:
    snapshot_scales = 4
    snapshots = solver.evaluator.add_file_handler(path + '/' + suffix + '/snapshots_{}'.format(label), sim_dt=0.0125, max_writes=5)
    snapshots.add_task(s, name='tracer', layout='g', scales=snapshot_scales)
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity', layout='g', scales=snapshot_scales)

max_timestep *= 0.5
logger.info('max_timestep = {}'.format(max_timestep))