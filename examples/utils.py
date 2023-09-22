import os, time
from collections import Counter
from pathlib import Path

import numpy as np

from examples.tesseract_planning_example_composer import OMPL_DEFAULT_NAMESPACE

TESSERACT_SUPPORT_DIR = os.environ["TESSERACT_RESOURCE_PATH"]
TESSERACT_TASK_COMPOSER_DIR = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]
TRAJOPT_DEFAULT_NAMESPACE = "TrajOptMotionPlannerTask"
os.environ["TRAJOPT_LOG_THRESH"] = "DEBUG"

from tesseract_robotics import tesseract_command_language as t_lang
from tesseract_robotics import tesseract_common as t_common
from tesseract_robotics.tesseract_command_language import ProfileDictionary
from tesseract_robotics.tesseract_environment import Environment
from tesseract_robotics.tesseract_motion_planners import (
    assignCurrentStateAsSeed,
    PlannerRequest,
    PlannerResponse,
)
from tesseract_robotics.tesseract_motion_planners_descartes import (
    ProfileDictionary_addProfile_DescartesPlanProfileD,
)
from tesseract_robotics.tesseract_motion_planners_ompl import (
    ProfileDictionary_addProfile_OMPLPlanProfile,
    OMPLDefaultPlanProfile,
    RRTConnectConfigurator,
    OMPLMotionPlanner,
)
from tesseract_robotics.tesseract_motion_planners_simple import (
    ProfileDictionary_addProfile_SimplePlannerPlanProfile,
    ProfileDictionary_addProfile_SimplePlannerCompositeProfile,
)
from tesseract_robotics import tesseract_motion_planners_trajopt as t_mpl_trajopt
from tesseract_robotics import tesseract_task_composer as t_task_comp

from tesseract_viewer_python.tesseract_robotics_viewer import TesseractViewer

task_composer_filename = os.environ["TESSERACT_TASK_COMPOSER_CONFIG_FILE"]


PROFILE_LUT = {
    # descartes
    "DescartesPlanProfileD": ProfileDictionary_addProfile_DescartesPlanProfileD,
    # ompl
    "OMPLPlanProfile": ProfileDictionary_addProfile_OMPLPlanProfile,
    # simple
    "SimplePlannerPlanProfile": ProfileDictionary_addProfile_SimplePlannerPlanProfile,
    "SimplePlannerCompositeProfile": ProfileDictionary_addProfile_SimplePlannerCompositeProfile,
    # trajopt
    "TrajOptSolverProfile": t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptSolverProfile,
    "TrajOptPlanProfile": t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptPlanProfile,
    "TrajOptCompositeProfile": t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptCompositeProfile,
    # task composer
    "CheckInputProfile": t_task_comp.ProfileDictionary_addProfile_CheckInputProfile,
    "ContactCheckProfile": t_task_comp.ProfileDictionary_addProfile_ContactCheckProfile,
    "FixStateBoundsProfile": t_task_comp.ProfileDictionary_addProfile_FixStateBoundsProfile,
    "FixStateCollisionProfile": t_task_comp.ProfileDictionary_addProfile_FixStateCollisionProfile,
    "IterativeSplineParameterizationProfile": t_task_comp.ProfileDictionary_addProfile_IterativeSplineParameterizationProfile,
    "MinLengthProfile": t_task_comp.ProfileDictionary_addProfile_MinLengthProfile,
    "ProfileSwitchProfile": t_task_comp.ProfileDictionary_addProfile_ProfileSwitchProfile,
    "RuckigTrajectorySmoothingCompositeProfile": t_task_comp.ProfileDictionary_addProfile_RuckigTrajectorySmoothingCompositeProfile,
    "RuckigTrajectorySmoothingMoveProfile": t_task_comp.ProfileDictionary_addProfile_RuckigTrajectorySmoothingMoveProfile,
    "TimeOptimalParameterizationProfile": t_task_comp.ProfileDictionary_addProfile_TimeOptimalParameterizationProfile,
    "UpsampleTrajectoryProfile": t_task_comp.ProfileDictionary_addProfile_UpsampleTrajectoryProfile,
}


def tesseract_task_composer_config_file():
    # OVERRIDE defaults that has no IPOPT trajopt
    # TODO
    config_path = t_common.FilesystemPath(
        "Y:\\CADCAM\\tesseract_planning\\tesseract_task_composer\\config\\task_composer_plugins.yaml"
    )
    return config_path


def support_dir(pth):
    return t_common.FilesystemPath(os.path.join(TESSERACT_SUPPORT_DIR, pth))


def compose_dir(pth):
    return t_common.FilesystemPath(os.path.join(TESSERACT_TASK_COMPOSER_DIR, pth))


def get_environment(url) -> tuple[Environment, t_common.ManipulatorInfo, list[str]]:
    """
    given a `url` load a URDF & SRDF and return an Enviornment and Manipulator instance and a
    list of joint names
    """
    locator = t_common.GeneralResourceLocator()
    env = Environment()
    # tesseract_support = os.environ["TESSERACT_SUPPORT_DIR"]
    urdf_path = locator.locateResource(f"{url}.urdf").getFilePath()
    srdf_path = locator.locateResource(f"{url}.srdf").getFilePath()

    urdf_path_str = t_common.FilesystemPath(urdf_path)
    srdf_path_str = t_common.FilesystemPath(srdf_path)

    assert env.init(urdf_path_str, srdf_path_str, locator)
    manip_info = t_common.ManipulatorInfo()
    manip_info.tcp_frame = "tool0"
    manip_info.manipulator = "manipulator"
    manip_info.working_frame = "base_link"
    joint_names = list(env.getJointGroup("manipulator").getJointNames())

    return env, manip_info, joint_names


def _translation(p) -> t_common.Isometry3d:
    H = np.eye(4)
    H[0:3, 3] = p
    return t_common.Isometry3d(H)


def _move_instruction_from_isometry3d(
    goal: t_common.Isometry3d, move_type=t_lang.MoveInstructionType_LINEAR
) -> t_lang.MoveInstructionPoly_wrap_MoveInstruction:
    cwp_cw = t_lang.CartesianWaypointPoly_wrap_CartesianWaypoint(
        t_lang.CartesianWaypoint(goal)
    )
    mip_mi = t_lang.MoveInstructionPoly_wrap_MoveInstruction(
        t_lang.MoveInstruction(cwp_cw, move_type, "CARTESIAN")
    )
    return mip_mi


def create_trajopt_profile_glass_example() -> t_lang.ProfileDictionary:
    profile = t_lang.ProfileDictionary()

    # todo: no ifopt
    composite_profile = t_mpl_trajopt.TrajOptDefaultCompositeProfile()

    composite_profile.collision_cost_config.enabled = True
    # todo: expected tc.CollisionEvaluatorType_DISCRETE_CONTINUOUS
    # composite_profile.collision_cost_config.type = tc.CollisionEvaluatorType_CONTINUOUS
    composite_profile.collision_cost_config.safety_margin = 0.01
    composite_profile.collision_cost_config.safety_margin_buffer = 0.01
    composite_profile.collision_cost_config.coeff = 1
    composite_profile.collision_constraint_config.enabled = True
    # todo: expected tc.CollisionEvaluatorType_DISCRETE_CONTINUOUS
    # composite_profile.collision_constraint_config.type = tc.CollisionEvaluatorType_CONTINUOUS
    composite_profile.collision_constraint_config.safety_margin = 0.01
    composite_profile.collision_constraint_config.safety_margin_buffer = 0.01
    composite_profile.collision_constraint_config.coeff = 1
    composite_profile.smooth_velocities = True
    composite_profile.smooth_accelerations = False
    composite_profile.smooth_jerks = False
    composite_profile.velocity_coeff = np.array([1.0])

    trajopt_solver_profile = t_mpl_trajopt.TrajOptDefaultSolverProfile()

    btr_params = t_mpl_trajopt.BasicTrustRegionSQPParameters()
    # btr_params.max_iter = 200
    btr_params.max_iter = 20000000
    btr_params.min_approx_improve = 1e-4
    btr_params.min_trust_box_size = 1e-2
    btr_params.trust_expand_ratio
    btr_params.log_dir = str(Path(__file__).parent)
    btr_params.log_results = True

    mt = t_mpl_trajopt.ModelType(t_mpl_trajopt.ModelType.QPOASES)  # 1.4 sec
    # mt = t_mpl_trajopt.ModelType(t_mpl_trajopt.ModelType.AUTO_SOLVER)

    # TODO: is it possible that this just wont solve without the t_mpl_trajopt.ModelType.BPMPD solver

    # seems to do its job: fails when I set gurobi; not build with that options
    # mt = ModelType(ModelType.GUROBI)

    trajopt_solver_profile.opt_info = btr_params
    trajopt_solver_profile.convex_solver = mt

    # TODO: add a method `add_profile` that add the correct profile
    # by inspecting composite_profile.__class__ to have something a little more
    # pythonic

    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptCompositeProfile(
        profile, TRAJOPT_DEFAULT_NAMESPACE, "UPRIGHT", composite_profile
    )

    plan_profile = t_mpl_trajopt.TrajOptDefaultPlanProfile()
    # plan_profile.joint_coeff = np.ones((7,))
    plan_profile.cartesian_coeff = np.array(
        [0.0, 0.0, 0.0, 5.0, 5.0, 5.0],
    )

    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptPlanProfile(
        profile, TRAJOPT_DEFAULT_NAMESPACE, "UPRIGHT", plan_profile
    )

    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptSolverProfile(
        profile, TRAJOPT_DEFAULT_NAMESPACE, "UPRIGHT", trajopt_solver_profile
    )

    return profile


def create_trajopt_profile_puzzle_example() -> t_lang.ProfileDictionary:
    # Create TrajOpt Profile
    trajopt_plan_profile = t_mpl_trajopt.TrajOptDefaultPlanProfile()
    trajopt_plan_profile.cartesian_coeff = np.array(
        [10, 10, 10, 10, 10, 0], dtype=np.float64
    )

    trajopt_composite_profile = t_mpl_trajopt.TrajOptDefaultCompositeProfile()
    trajopt_composite_profile.collision_constraint_config.enabled = False
    trajopt_composite_profile.collision_cost_config.enabled = True
    trajopt_composite_profile.collision_cost_config.safety_margin = 0.025
    trajopt_composite_profile.collision_cost_config.type = (
        t_mpl_trajopt.CollisionEvaluatorType_SINGLE_TIMESTEP
    )
    trajopt_composite_profile.collision_cost_config.coeff = 20

    trajopt_solver_profile = t_mpl_trajopt.TrajOptDefaultSolverProfile()

    btr_params = t_mpl_trajopt.BasicTrustRegionSQPParameters()
    btr_params.max_iter = 200
    btr_params.max_iter = 20000000
    btr_params.min_approx_improve = 1e-6
    btr_params.min_trust_box_size = 1e-3

    mt = t_mpl_trajopt.ModelType(t_mpl_trajopt.ModelType.OSQP)
    # seems to do its job: fails when I set gurobi; not build with that options
    # mt = ModelType(ModelType.GUROBI)

    trajopt_solver_profile.opt_info = btr_params
    trajopt_solver_profile.convex_solver = mt

    # Create profile dictionary
    trajopt_profiles = t_lang.ProfileDictionary()
    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptPlanProfile(
        trajopt_profiles, TRAJOPT_DEFAULT_NAMESPACE, "CARTESIAN", trajopt_plan_profile
    )

    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptCompositeProfile(
        trajopt_profiles,
        TRAJOPT_DEFAULT_NAMESPACE,
        "DEFAULT",
        trajopt_composite_profile,
    )

    t_mpl_trajopt.ProfileDictionary_addProfile_TrajOptSolverProfile(
        trajopt_profiles, TRAJOPT_DEFAULT_NAMESPACE, "DEFAULT", trajopt_solver_profile
    )
    return trajopt_profiles


class TesseractPlanner:
    def __init__(
        self,
        joint_names: list[str],
        mi: t_common.ManipulatorInfo,
        t_env: Environment,
    ):
        self.joint_names = joint_names
        # Create Program

        self.manip_info = mi
        self.t_env = t_env
        self.task_composer = TesseractTaskComposer(self)
        self.profile = t_lang.ProfileDictionary()

        self.program = t_lang.CompositeInstruction(
            "DEFAULT", t_lang.CompositeInstructionOrder_ORDERED
        )
        self.program.setManipulatorInfo(self.manip_info)
        # self.task_composer.add_program(self.program, self.manip_info)

        # self.joint_names = [i for i in self.t_env.getJointNames()]

        self._cartesian_point_counter = 0

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, value: ProfileDictionary):
        self._profile = value
        self.task_composer.profile = value

    def update_env(self):
        # Get the state solver. This must be called again after environment is updated
        solver = self.t_env.getStateSolver()

        # Get the discrete contact manager. This must be called again after the environment is updated
        manager = self.t_env.getDiscreteContactManager()
        manager.setActiveCollisionObjects(self.t_env.getActiveLinkNames())

    def set_viewer(self, viewer: TesseractViewer):
        is_viewer = isinstance(viewer, TesseractViewer)
        if not is_viewer:
            raise TypeError(
                f"expected a TesseractViewer instance, got a {viewer} instead"
            )
        else:
            self.viewer = viewer

    def add_cartesian_poses(self, tool_poses: list[t_common.Isometry3d]):
        # Create cartesian waypoint
        for n, i in enumerate(tool_poses):
            plan_instruction = _move_instruction_from_isometry3d(i)
            plan_instruction.setDescription(
                f"cartesian_waypoint_{self._cartesian_point_counter:05d}"
            )
            self._cartesian_point_counter += 1
            self.program.appendMoveInstruction(plan_instruction)

    def add_joint_pose(
        self,
        joint_pose,
        group_name="JOINT_DEFAULT",
        move_type=t_lang.MoveInstructionType_LINEAR,
        description="go joints",
    ):
        waypoint = t_lang.StateWaypoint(self.joint_names, joint_pose)
        wp = t_lang.StateWaypointPoly_wrap_StateWaypoint(waypoint)

        joint_waypoint_instruction = t_lang.MoveInstruction(wp, move_type, group_name)
        joint_waypoint_instruction.setDescription(description)

        mipmi = t_lang.MoveInstructionPoly_wrap_MoveInstruction(
            joint_waypoint_instruction
        )
        self.program.appendMoveInstruction(mipmi)

    def add_to_profile(
        self,
        profile_to_add,
        profile_name: str = "DEFAULT",
        namespace: str = "DEFAULT_NS",
    ):
        p = profile_to_add.__class__.__name__.split("ProfileDictionary_addProfile_")[1]
        assert p in PROFILE_LUT
        profile_dictionary_add_profile = PROFILE_LUT[p]

        profile_dictionary_add_profile(
            self.profile_dict, TRAJOPT_DEFAULT_NAMESPACE, profile_name, profile_to_add
        )

    def as_joint_trajectory(self) -> t_common.JointTrajectory:
        # Plot Process Trajectory
        # TODO as composite instruction

        output_key = self.task_composer.task.getOutputKeys()[0]
        _ci: t_common.AnyPoly = self.task_composer.task_data.getData(output_key)

        ci = t_lang.AnyPoly_as_CompositeInstruction(_ci)

        trajectory: t_common.JointTrajectory = t_lang.toJointTrajectory(ci)
        return trajectory

    def print_joints(self, jt: t_common.JointTrajectory):
        # Display the output
        # Print out the resulting waypoints
        for instr in jt:
            assert instr.isMoveInstruction()
            move_instr1 = t_lang.InstructionPoly_as_MoveInstructionPoly(instr)
            wp1 = move_instr1.getWaypoint()
            assert wp1.isStateWaypoint()
            wp = t_lang.WaypointPoly_as_StateWaypointPoly(wp1)
            print(f"Joint Positions: {wp.getPosition().flatten()} time: {wp.getTime()}")

    def create_viewer(self, update_env=False) -> TesseractViewer:
        # Create a viewer and set the environment so the results can be displayed later

        if self.viewer:
            print("viewer already created")
            return self.viewer
        else:
            viewer = TesseractViewer()
            viewer.update_environment(self.t_env, [0, 0, 0])

            jnts = self.t_env.getCurrentJointValues()

            if update_env:
                self.update_env()

            # Set the initial state of the robot
            viewer.update_joint_positions(
                # self.joint_names, np.array([1, -0.2, 0.01, 0.3, -0.5, 1])
                self.joint_names,
                jnts,
            )

            # Start the viewer
            viewer.start_serve_background()
        return viewer

    def update_viewer(self, viewer, jt: t_common.JointTrajectory):
        # Update the viewer with the results to animate the trajectory
        # Open web browser to http://localhost:8000 to view the results
        viewer.update_trajectory(jt)
        viewer.plot_trajectory(jt, self.manip_info)

    def plot_trajectory(self, is_aborted, is_succesful, viewer=None):
        if not is_aborted and is_succesful:
            jt = self.as_joint_trajectory()
            self.print_joints(jt)

            if not viewer:
                self.viewer = self.create_viewer()
            else:
                self.set_viewer(viewer)

            self.print_joints(jt)
            self.update_viewer(self.viewer, jt)

    def plan(self) -> tuple[bool, bool]:
        # Assign the current state as the seed for cartesian waypoints
        assignCurrentStateAsSeed(self.program, self.t_env)
        self.task_composer.add_program(self.program, self.manip_info)

        is_aborted, is_successful = self.task_composer.run()

        return is_aborted, is_successful

    def plan_ompl(self):
        # Initialize the OMPL planner for RRTConnect algorithm
        plan_profile = OMPLDefaultPlanProfile()
        plan_profile.planners.clear()
        plan_profile.planners.append(RRTConnectConfigurator())

        # Create the profile dictionary. Profiles can be used to customize the behavior of the planner. The module
        # level function `ProfileDictionary_addProfile_OMPLPlanProfile` is used to add a profile to the dictionary. All
        # profile types have associated profile dictionary functions.
        profiles = ProfileDictionary()
        ProfileDictionary_addProfile_OMPLPlanProfile(
            profiles, OMPL_DEFAULT_NAMESPACE, "TEST_PROFILE", plan_profile
        )

        cur_state = self.t_env.getState()

        # Create the planning request and run the planner
        request = PlannerRequest()
        request.instructions = self.program
        request.env = self.t_env
        request.env_state = cur_state
        request.profiles = profiles

        ompl_planner = OMPLMotionPlanner(OMPL_DEFAULT_NAMESPACE)

        response: PlannerResponse = ompl_planner.solve(request)
        assert response.successful
        results_instruction = response.results

        print("solved OMPL")
        return response


class TesseractTaskComposer:  # TODO: poorly named nothing better comes to mind
    def __init__(self, planner: TesseractPlanner):
        self.planner = planner  # reference to the parent class
        fs_pth = tesseract_task_composer_config_file()
        self.factory = t_task_comp.TaskComposerPluginFactory(fs_pth)
        self.profile = t_lang.ProfileDictionary()

    def add_program(
        self,
        program: t_lang.CompositeInstruction,
        manip_info: t_common.ManipulatorInfo,
    ):
        # Create the task composer node
        # In this case the FreespacePipeline is used
        # Many others are available.
        # TODO!
        self.task = self.factory.createTaskComposerNode("TrajOptPipeline")
        input_key = self.task.getInputKeys()[0]
        output_key = self.task.getOutputKeys()[0]

        # Create an AnyPoly containing the program. This explicit step is required because the Python bindings do not
        # support implicit conversion from the CompositeInstruction to the AnyPoly
        self.program_anypoly = t_lang.AnyPoly_wrap_CompositeInstruction(program)

        self.task_data = t_task_comp.TaskComposerDataStorage()
        self.task_data.setData(input_key, self.program_anypoly)

        # Create the task problem and input
        task_planning_problem = t_task_comp.PlanningTaskComposerProblemUPtr.make_unique(
            self.planner.t_env, self.task_data, self.profile
        )

        task_problem = (
            t_task_comp.PlanningTaskComposerProblemUPtr_as_TaskComposerProblemUPtr(
                task_planning_problem
            )
        )
        self.task_input = t_task_comp.TaskComposerInput(task_problem)

        # Create an executor to run the task
        self.task_executor = self.factory.createTaskComposerExecutor("TaskflowExecutor")

    def run(self) -> tuple[bool, bool]:
        assert hasattr(self, "task_executor"), "call create_task_executor"

        start = time.time()
        # Run the task and wait for completion
        future = self.task_executor.run(self.task.get(), self.task_input)
        future.wait()
        stop = time.time() - start

        is_aborted = self.task_input.isAborted()
        print(f"was planning is_aborted? {is_aborted}")
        is_successful = self.task_input.isSuccessful()
        print(f"was planning succesful? {is_successful}")

        print(f"planning took {stop} seconds")
        return is_aborted, is_successful
