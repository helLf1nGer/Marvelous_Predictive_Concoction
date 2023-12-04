"""Set up the zeroth-order QP problem for stance leg control.

For details, please refer to section XX of this paper:
https://arxiv.org/abs/2009.10019
"""

import numpy as np
# import numba
# import quadprog  # pytype:disable=import-error
from pyomo.environ import ConcreteModel, Var, Objective, Constraint, minimize, NonNegativeReals, SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

np.set_printoptions(precision=3, suppress=True)

ACC_WEIGHT = np.array([1., 1., 1., 10., 10, 1.])


# @numba.jit(nopython=True, parallel=True, cache=True)
def compute_mass_matrix(robot_mass, robot_inertia, foot_positions):
    # yaw = 0.  # Set yaw to 0 for now as all commands are local.
    # rot_z = np.array([[np.cos(yaw), np.sin(yaw), 0.],
    #                   [-np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    rot_z = np.eye(3)

    inv_mass = np.eye(3) / robot_mass
    inv_inertia = np.linalg.inv(robot_inertia)
    mass_mat = np.zeros((6, 12))

    for leg_id in range(4):
        mass_mat[:3, leg_id * 3:leg_id * 3 + 3] = inv_mass

        x = foot_positions[leg_id]
        foot_position_skew = np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]],
                                       [-x[1], x[0], 0]])
        mass_mat[3:6, leg_id * 3:leg_id * 3 +
                 3] = rot_z.T.dot(inv_inertia).dot(foot_position_skew)
    return mass_mat

# @numba.jit(nopython=True, parallel=True, cache=True)


def compute_constraint_matrix(mpc_body_mass,
                              contacts,
                              friction_coef=0.8,
                              f_min_ratio=0.1,
                              f_max_ratio=10):
    f_min = f_min_ratio * mpc_body_mass * 9.8
    f_max = f_max_ratio * mpc_body_mass * 9.8
    A = np.zeros((24, 12))
    lb = np.zeros(24)
    for leg_id in range(4):
        A[leg_id * 2, leg_id * 3 + 2] = 1
        A[leg_id * 2 + 1, leg_id * 3 + 2] = -1
        if contacts[leg_id]:
            lb[leg_id * 2], lb[leg_id * 2 + 1] = f_min, -f_max
        else:
            lb[leg_id * 2] = -1e-7
            lb[leg_id * 2 + 1] = -1e-7

    # Friction constraints
    for leg_id in range(4):
        row_id = 8 + leg_id * 4
        col_id = leg_id * 3
        lb[row_id:row_id + 4] = np.array([0, 0, 0, 0])
        A[row_id, col_id:col_id + 3] = np.array([1, 0, friction_coef])
        A[row_id + 1, col_id:col_id + 3] = np.array([-1, 0, friction_coef])
        A[row_id + 2, col_id:col_id + 3] = np.array([0, 1, friction_coef])
        A[row_id + 3, col_id:col_id + 3] = np.array([0, -1, friction_coef])
    return A.T, lb


# @numba.jit(nopython=True, cache=True)
def compute_objective_matrix(mass_matrix, desired_acc, acc_weight, reg_weight):
    g = np.array([0., 0., 9.8, 0., 0., 0.])
    Q = np.diag(acc_weight)
    R = np.ones(12) * reg_weight

    pre_quad_term = mass_matrix.T.dot(Q).dot(mass_matrix)
    quad_term = (pre_quad_term + pre_quad_term.T) / \
        2 + np.eye(12) * R  # Ensure symmetry
    linear_term = 1 * (g + desired_acc).T.dot(Q).dot(mass_matrix)
    return quad_term, linear_term


def compute_contact_force(robot, desired_acc, contacts, acc_weight=ACC_WEIGHT, reg_weight=1e-4, friction_coef=0.45, f_min_ratio=0.1, f_max_ratio=10.):
    mass_matrix = compute_mass_matrix(robot.MPC_BODY_MASS, np.array(
        robot.MPC_BODY_INERTIA).reshape((3, 3)), robot.GetFootPositionsInBaseFrame())
    G, a = compute_objective_matrix(
        mass_matrix, desired_acc, acc_weight, reg_weight)
    C, b = compute_constraint_matrix(
        robot.MPC_BODY_MASS, contacts, friction_coef, f_min_ratio, f_max_ratio)
    G += 1e-4 * np.eye(12)

    model = ConcreteModel()
    model.forces = Var(range(12))

    # Corrected Objective
    def obj_expression(model):
        return 0.5 * sum(G[i, j] * model.forces[i] * model.forces[j] for i in range(12) for j in range(12)) + sum(a[i] * model.forces[i] for i in range(12))

    model.objective = Objective(rule=obj_expression, sense=minimize)

    # Constraints
    def con_rule(model, i):
        return sum(C[i, j] * model.forces[j] for j in range(12)) >= b[i]

    model.constraints = Constraint(range(24), rule=con_rule)

    solver = SolverFactory('ipopt')
    solution = solver.solve(model)

    if (solution.solver.status == SolverStatus.ok) and (solution.solver.termination_condition == TerminationCondition.optimal):
        return np.array([-model.forces[i].value for i in range(12)]).reshape((4, 3))
    else:
        print("Solver failed to find optimal solution")
        return None
