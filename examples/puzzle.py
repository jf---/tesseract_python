import os, sys

# TODO
# Seems to me that a `Vector3d` class isn't available from tesseract?
from compas.geometry import Vector

from examples.utils import create_trajopt_profile_puzzle_example, TesseractPlanner

# this example is transliterated from C++, you'll find the original here:
# https://github.com/tesseract-robotics/tesseract_planning/blob/master/tesseract_examples/src/puzzle_piece_example.cpp
# For some context of this particular example, see ROSCon 2018 Madrid: Optimization Motion
# Planning with Tesseract and TrajOpt for Industrial Applications
# https://vimeo.com/293314190 @ 4:27

# TODO
sys.path.extend(
    ["Y:\CADCAM\tesseract_python\tesseract_viewer_python\tesseract_robotics_viewer"]
)

from tesseract_viewer_python.tesseract_robotics_viewer import TesseractViewer

# TODO
os.environ["TRAJOPT_LOG_THRESH"] = "DEBUG"

import numpy as np
from tesseract_robotics.tesseract_common import (
    Isometry3d,
    ManipulatorInfo,
    GeneralResourceLocator,
)

from utils import get_environment


def make_puzzle_tool_poses() -> list[Isometry3d]:
    path = []  # results
    locator = GeneralResourceLocator()
    # Locate the CSV file using the provided resource locator
    resource = locator.locateResource(
        "package://tesseract_support/urdf/puzzle_bent.csv"
    )
    file_path = resource.getFilePath()

    # Open the CSV file for reading
    with open(file_path, "r") as indata:
        lines = indata.readlines()

    for lnum, line in enumerate(lines):
        if lnum < 2:
            continue

        cells = line.split(",")
        xyzijk = [float(cell) for cell in cells[1:]]  # Ignore the first value

        pos = Vector(xyzijk[0], xyzijk[1], xyzijk[2])  # Convert from mm to meters
        pos /= 1000

        print(pos)
        norm = Vector(xyzijk[3], xyzijk[4], xyzijk[5])

        norm.unitize()

        temp_x = (pos * -1).unitized()
        y_axis = norm.cross(temp_x).unitized()
        x_axis = y_axis.cross(norm).unitized()

        # Create an Isometry3d pose
        pose = Isometry3d()
        mat = pose.matrix()

        mat[0][:3] = x_axis
        mat[1][:3] = y_axis
        mat[2][:3] = norm
        mat[3][:3] = pos

        path.append(pose)

    return path


# path = ResourceLocator()


def create_manip() -> ManipulatorInfo:
    # Create manipulator information for the program
    mi = ManipulatorInfo()
    mi.manipulator = "manipulator"
    mi.working_frame = "part"
    mi.tcp_frame = "grinder_frame"
    return mi


class PuzzlePieceExample:
    def __init__(self, env, plotter=None):
        self.env = env
        self.plotter = plotter

    def run(self):
        if self.plotter is not None:
            self.plotter.waitForConnection()

        # Set the robot initial state
        joint_names = [
            "joint_a1",
            "joint_a2",
            "joint_a3",
            "joint_a4",
            "joint_a5",
            "joint_a6",
            "joint_a7",
        ]
        joint_pos = [-0.785398, 0.4, 0.0, -1.9, 0.0, 1.0, 0.0]

        mi = create_manip()

        # Get Tool Poses
        tool_poses = make_puzzle_tool_poses()

        profile_dict = create_trajopt_profile_puzzle_example()

        pl = TesseractPlanner(joint_names, mi, env)
        pl.profile_dict = profile_dict
        pl.t_env.setState(joint_names, np.array(joint_pos))
        pl.update_env()

        pl.add_cartesian_poses(tool_poses)

        pl.task_composer.add_program(pl.program, pl.manip_info)

        is_aborted, is_successful = pl.plan()

        pl.plot_trajectory(is_aborted, is_successful)


if __name__ == "__main__":
    make_puzzle_tool_poses()
    env, manip_info, joint_names = get_environment(
        "package://tesseract_support/urdf/puzzle_piece_workcell"
    )

    # Create a viewer and set the environment so the results can be displayed later
    viewer = TesseractViewer()
    viewer.update_environment(env, [0, 0, 0])

    # Set the initial state of the robot
    viewer.update_joint_positions(joint_names, np.ones(len(joint_names)) * 0.1)

    # Start the viewer
    viewer.start_serve_background()

    ppe = PuzzlePieceExample(env)
    results = ppe.run()

    input("press to exit the viewer ( http://localhost:8000 )")
