"""Example of whole body controller on A1 robot."""
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.robots.gamepad import gamepad_reader
from motion_imitation.robots import robot_config
from motion_imitation.robots import a1
from mpc_controller import torque_stance_leg_controller_quadprog as torque_stance_leg_controller
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import locomotion_controller
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import com_velocity_estimator
import pybullet  # pytype:disable=import-error
from pybullet_utils import bullet_client
import pybullet_data
import time
import scipy.interpolate
import numpy as np
from datetime import datetime
from absl import logging
from absl import flags
from absl import app

# from mpc_controller import torque_stance_leg_controller
# import mpc_osqp


flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_string("video_filepath", None,
                    "If set, save a video here. Should end with .mp4. Requires moviepy.")
flags.DEFINE_bool("use_gamepad", False,
                  "whether to use gamepad to provide control input.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_bool("gui_server", False, "Whether to connect to an existing pybullet GUI_SERVER.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")
flags.DEFINE_bool("bumpy_terrain", False,
                  "whether to use bumpy or flat terrain.")
flags.DEFINE_string("solver", "quadprog",
                    "The QP solver to use. Options are 'quadprog', 'OSQP', or any other CVXPY solvers .")
FLAGS = flags.FLAGS

VIDEO_FPS = 50
VIDEO_TIMESTEP = 1 / VIDEO_FPS
VIDEO_DIMENSIONS = (480, 360)

_STANCE_DURATION_SECONDS = [
    0.3
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).

# Standing
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.SWING,
# )

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _generate_example_linear_angular_speed(t):
    """Creates an example speed profile based on time for demo purpose."""
    vx = 0.6
    vy = 0.2
    wz = 0.8

    time_points = (0, 5, 10, 15, 20, 25, 30)
    speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                    (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

    speed = scipy.interpolate.interp1d(time_points,
                                       speed_points,
                                       kind="previous",
                                       fill_value="extrapolate",
                                       axis=0)(t)

    return speed[0:3], speed[3], False


def _setup_controller(robot, solver_name):
    """Demonstrates how to create a locomotion controller."""
    desired_speed = (0, 0)
    desired_twisting_speed = 0

    gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
        robot,
        stance_duration=_STANCE_DURATION_SECONDS,
        duty_factor=_DUTY_FACTOR,
        initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
        initial_leg_state=_INIT_LEG_STATE)
    window_size = 20 if not FLAGS.use_real_robot else 1
    state_estimator = com_velocity_estimator.COMVelocityEstimator(
        robot, window_size=window_size)
    sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_height=robot.MPC_BODY_HEIGHT,
        foot_clearance=0.01)

    st_controller = torque_stance_leg_controller.TorqueStanceLegController(
        robot,
        gait_generator,
        state_estimator,
        desired_speed=desired_speed,
        desired_twisting_speed=desired_twisting_speed,
        desired_body_height=robot.MPC_BODY_HEIGHT,
        solver_name=solver_name
        # ,qp_solver = mpc_osqp.QPOASES #or mpc_osqp.OSQP
    )

    print("Solver Chosen: ", solver_name)
    controller = locomotion_controller.LocomotionController(
        robot=robot,
        gait_generator=gait_generator,
        state_estimator=state_estimator,
        swing_leg_controller=sw_controller,
        stance_leg_controller=st_controller,
        clock=robot.GetTimeSinceReset)
    return controller


def _update_controller_params(controller, lin_speed, ang_speed):
    controller.swing_leg_controller.desired_speed = lin_speed
    controller.swing_leg_controller.desired_twisting_speed = ang_speed
    controller.stance_leg_controller.desired_speed = lin_speed
    controller.stance_leg_controller.desired_twisting_speed = ang_speed


def load_bumpy_terrain(pybullet_client, using_gui):
    n_rows = 128
    n_cols = 128
    heightfield_data = [0] * n_rows * n_cols
    max_height_perturbation = 0.05
    for j in range(int(n_rows / 2)):
        for i in range(int(n_cols / 2)):
            height = np.random.uniform(0, max_height_perturbation)
            heightfield_data[2 * i + 2 * j * n_rows] = height
            heightfield_data[2 * i + 1 + 2 * j * n_rows] = height
            heightfield_data[2 * i + (2 * j + 1) * n_rows] = height
            heightfield_data[2 * i + 1 + (2 * j + 1) * n_rows] = height

    # Rendering while loading is slow.
    if using_gui:
        pybullet_client.configureDebugVisualizer(
            pybullet_client.COV_ENABLE_RENDERING, 0)

    terrain_shape = pybullet_client.createCollisionShape(
        shapeType=pybullet_client.GEOM_HEIGHTFIELD,
        flags=pybullet_client.GEOM_CONCAVE_INTERNAL_EDGE,
        meshScale=[.15, .15, 1],
        heightfieldData=heightfield_data,
        numHeightfieldRows=n_rows,
        numHeightfieldColumns=n_cols,
        replaceHeightfieldIndex=-1)
    terrain = pybullet_client.createMultiBody(0, terrain_shape)
    texture_id = pybullet_client.loadTexture("checker_blue.png")
    pybullet_client.changeVisualShape(
        terrain, -1, textureUniqueId=texture_id, rgbaColor=(1, 1, 1, 1))
    if using_gui:
        pybullet_client.configureDebugVisualizer(
            pybullet_client.COV_ENABLE_RENDERING, 1)


def main(argv):
    """Runs the locomotion controller example."""
    del argv  # unused
    if FLAGS.video_filepath:
        import moviepy.editor
        directory = os.path.dirname(FLAGS.video_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Construct simulator
    if FLAGS.show_gui and not FLAGS.use_real_robot:
        p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
    elif FLAGS.gui_server:
        p = bullet_client.BulletClient(connection_mode=pybullet.SHARED_MEMORY)
        p.resetSimulation()
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-30,
                                     cameraTargetPosition=(0, 0, 0))
    else:
        p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
    p.setPhysicsEngineParameter(numSolverIterations=30)
    p.setTimeStep(0.001)
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(enableConeFriction=0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if FLAGS.bumpy_terrain:
        load_bumpy_terrain(p, FLAGS.show_gui)
    else:
        p.loadURDF("plane.urdf")

    # Construct robot class:
    if FLAGS.use_real_robot:
        from motion_imitation.robots import a1_robot
        robot = a1_robot.A1Robot(
            pybullet_client=p,
            motor_control_mode=robot_config.MotorControlMode.HYBRID,
            enable_action_interpolation=False,
            time_step=0.002,
            action_repeat=1)
    else:
        robot = a1.A1(p,
                      motor_control_mode=robot_config.MotorControlMode.HYBRID,
                      enable_action_interpolation=False,
                      reset_time=2,
                      time_step=0.002,
                      action_repeat=1)

    controller = _setup_controller(robot, FLAGS.solver)

    controller.reset()
    if FLAGS.use_gamepad:
        gamepad = gamepad_reader.Gamepad()
        command_function = gamepad.get_command
    else:
        command_function = _generate_example_linear_angular_speed

    if FLAGS.logdir:
        logdir = os.path.join(FLAGS.logdir,
                              datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        os.makedirs(logdir)

    start_time = robot.GetTimeSinceReset()
    current_time = start_time
    com_vels, imu_rates, actions = [], [], []
    video_frames = []
    while current_time - start_time < FLAGS.max_time_secs:
        start_time_robot = current_time
        start_time_wall = time.time()
        # Updates the controller behavior parameters.
        lin_speed, ang_speed, e_stop = command_function(current_time)
        if e_stop:
            logging.info("E-stop kicked, exiting...")
            break
        _update_controller_params(controller, lin_speed, ang_speed)
        controller.update()
        hybrid_action, _ = controller.get_action()
        com_vels.append(np.array(robot.GetBaseVelocity()).copy())
        imu_rates.append(np.array(robot.GetBaseRollPitchYawRate()).copy())
        actions.append(hybrid_action)
        robot.Step(hybrid_action)
        current_time = robot.GetTimeSinceReset()
        if FLAGS.video_filepath and current_time >= len(video_frames) * VIDEO_TIMESTEP:
          img = p.getCameraImage(*VIDEO_DIMENSIONS, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2][:,:,:3]
          video_frames.append(img)

        if not FLAGS.use_real_robot:
            expected_duration = current_time - start_time_robot
            actual_duration = time.time() - start_time_wall
            if actual_duration < expected_duration:
                time.sleep(expected_duration - actual_duration)
    if FLAGS.use_gamepad:
        gamepad.stop()

    if FLAGS.logdir:
        np.savez(os.path.join(logdir, 'action.npz'),
                 action=actions,
                 com_vels=com_vels,
                 imu_rates=imu_rates)
        logging.info("logged to: {}".format(logdir))
    if len(video_frames):
      clip = moviepy.editor.ImageSequenceClip(video_frames, fps=VIDEO_FPS)
      clip.write_videofile(FLAGS.video_filepath)


if __name__ == "__main__":
    app.run(main)
