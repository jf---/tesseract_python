#  ported from file glass_upright_example.cpp
# for the original see:
#  https://github.com/tesseract-robotics/tesseract_planning/blob/master/tesseract_examples/src/glass_upright_example.cpp

import numpy as np
from tesseract_robotics import tesseract_environment as te
from tesseract_robotics import tesseract_geometry as tg
from tesseract_robotics import tesseract_scene_graph as tsg
from tesseract_robotics.tesseract_common import (
    Isometry3d,
    ManipulatorInfo,
    JointTrajectory,
)
from tesseract_robotics.tesseract_environment import Environment

from examples.utils import (
    get_environment,
    create_trajopt_profile_glass_example,
    TesseractPlanner,
)


class GlassUprightExample:
    def __init__(self, env: Environment, manip_info: ManipulatorInfo, debug: bool):
        manip_info.manipulator = "manipulator"
        manip_info.tcp_frame = "tool0"
        manip_info.working_frame = "base_link"

        self.debug = debug

        (
            self.joint_names,
            self.joint_start_pos,
            self.joint_end_pos,
        ) = self._initial_joint_state()
        self.planner = TesseractPlanner(self.joint_names, manip_info, env)
        self.planner.t_env.setState(self.joint_names, self.joint_start_pos)
        # self.planner.update_env()

    def add_sphere(self) -> te.AddLinkCommand:
        link_sphere = tsg.Link("sphere_attached")
        visual = tsg.Visual()
        origin = Isometry3d.Identity()

        ll = np.array([0.5, 0, 0.55])
        # TODO: stange that a `Translation3d` is not an accepted argument
        origin.setTranslation(ll)

        sphere = tg.Sphere(0.15)
        visual.geometry = sphere
        link_sphere.visual.push_back(visual)

        collision = tsg.Collision()
        collision.origin = visual.origin
        collision.geometry = visual.geometry
        link_sphere.collision.push_back(collision)

        joint_sphere = tsg.Joint("joint_sphere_attached")
        joint_sphere.parent_link_name = "base_link"
        joint_sphere.child_link_name = link_sphere.getName()
        joint_sphere.type = tsg.JointType_FIXED

        cmd = te.AddLinkCommand(link_sphere, joint_sphere)

        return cmd

    def _initial_joint_state(self):
        #   // Set the robot initial state
        joint_names = []
        joint_names.append("joint_a1")
        joint_names.append("joint_a2")
        joint_names.append("joint_a3")
        joint_names.append("joint_a4")
        joint_names.append("joint_a5")
        joint_names.append("joint_a6")
        joint_names.append("joint_a7")

        joint_start_pos = np.zeros((7,))
        joint_end_pos = np.zeros((7,))

        joint_start_pos[0] = -0.4
        joint_start_pos[1] = 0.2762
        joint_start_pos[2] = 0.0
        joint_start_pos[3] = -1.3348
        joint_start_pos[4] = 0.0
        joint_start_pos[5] = 1.4959
        joint_start_pos[6] = 0.0
        #
        #   Eigen::VectorXd self.joint_end_pos(7);
        joint_end_pos[0] = 0.4
        joint_end_pos[1] = 0.2762
        joint_end_pos[2] = 0.0
        joint_end_pos[3] = -1.3348
        joint_end_pos[4] = 0.0
        joint_end_pos[5] = 1.4959
        joint_end_pos[6] = 0.0

        return joint_names, joint_start_pos, joint_end_pos

    def plan(self):
        cmd = self.add_sphere()
        self.planner.t_env.applyCommand(cmd)
        self.planner.update_env()

        profile = create_trajopt_profile_glass_example()
        self.planner.profile = profile

        self.planner.add_joint_pose(self.joint_start_pos, group_name="UPRIGHT")
        self.planner.add_joint_pose(self.joint_end_pos, group_name="UPRIGHT")

        self.planner.task_composer.add_program(
            self.planner.program, self.planner.manip_info
        )

        self.planner.program.setProfile("UPRIGHT")
        # self.planner.create_viewer()

        is_aborted, is_successful = self.planner.plan()
        self.planner.plot_trajectory(is_aborted, is_successful)
        # response = self.planner.plan_ompl()

        return is_aborted, is_successful


def run():
    env, manip_info, joint_names = get_environment(
        "package://tesseract_support/urdf/lbr_iiwa_14_r820"
    )


    print(env, manip_info)

    gue = GlassUprightExample(env, manip_info, True)
    gue.plan()

    input("press enter to exit")


if __name__ == "__main__":
    run()
