# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone scoring script for Barkour.

Reads a csv containing timestamps and (x, y) positions of the robot's base, and
outputs the score the robot received on the Barkour obstacle course.

python score_barkour.py --log_file=data/barkour_run_0.csv
> 0.877
python score_barkour.py --log_file=data/barkour_run_0.csv --touched_jump
> 0.753
python score_barkour.py --log_file=data/barkour_run_1.csv
> 0.854
python score_barkour.py --log_file=data/barkour_run_2.csv
> 0.020
"""

import collections
import dataclasses
import math
import pprint

from absl import app
from absl import flags
import numpy as np


log_file_flag = flags.DEFINE_string(
    "log_file",
    None,
    "Location of file containing timestamps and robot positions.",
)
touched_jumpboard_flag = flags.DEFINE_bool(
    "touched_jump",
    False,
    "Whether robot made contact with the broad jump, failing that obstacle.",
)
# The distance to the gate at which the gate counts as reached.
SUCCESS_DISTANCE = 0.02  # meters
# Time penalty applied if the robot doesn't reach the end table.
MAX_TIME = 50  # seconds
PENALTY_PER_OBSTACLE = 0.1
# Nominal size in meters. This is the approximate length of a trajectory
# completing the obstacle, including approach distance.
NOMINAL_SIZE = {
    "start table": 1,
    "weave poles": 6,
    "a frame": 6,
    "jump": 4,
    "end table": 1,
}
TARGET_VELOCITY = 1.69  # meters/second
# Expected completion time for each obstacle.
ALLOTED_TIMES = {k: v / TARGET_VELOCITY for k, v in NOMINAL_SIZE.items()}
# The position (x,y) of the end table that the robot has to reach and stand on.
END_TABLE_POSITION = np.array((4.5, -4.5))
END_TABLE_HALF_WIDTH = 0.5
# Amount of time robot must stay on end table to complete course.
END_TABLE_COMPLETION_TIME = 5


# Each obstacle is composed of gates that the agent has to go through.
@dataclasses.dataclass(frozen=True)
class Gate:
  # (x, y) coordinates in meters
  left_pole: np.ndarray
  right_pole: np.ndarray

OBSTACLES_DICT = collections.OrderedDict([
    ("start table", [Gate([1.4, 0], [1.4, -0.7])]),
    (
        "weave poles",
        [
            Gate([1.7, 0], [1.7, -0.7]),
            Gate([2.5, -0.5], [1.7, -0.7]),
            Gate([2.5, -0.5], [3.2, -0.7]),
            Gate([3.8, -0.5], [3.2, -0.7]),
            Gate([3.8, -0.5], [4.3, -0.7]),
            Gate([5, -0.7], [4.3, -0.7]),
        ],
    ),
    (
        "a frame",
        [
            Gate([4.35, -2.0], [3.65, -1.1]),
            Gate([2.92, -3.0], [2.22, -2.1]),
            Gate([1.5, -4.0], [0.8, -3.1]),
        ],
    ),
    ("jump", [Gate([1.7, -4], [1.7, -5]), Gate([3.7, -4], [3.7, -5])]),
    ("end table", [Gate([4.1, -4], [4.1, -5])]),
])


def score_barkour(timestamps, positions, touched_jump):
  """Computes overall score and metrics for Barkour obstacle course.

  Args:
    timestamps: Sequence of timestamps (seconds) corresponding to positions. Can
      start at any value, as only differences are considered.
    positions: Array of (x, y) positions (meters) of the robot's base, relative
      to the Barkour scene. These will be checked against the positions of the
      Gates that make up the obstacles.
    touched_jump: Boolean indicating whether the robot touched the broad jump
      obstacle. If so, that obstacle counts as failed, even if the robot
      successfully completes the Gates.

  Returns:
    score: Barkour score in the range [0, 1].
    metrics: Dictionary containing extra information on the run.
  """
  completed_gates = {obs: 0 for obs in OBSTACLES_DICT}
  time_reached_end_table = 0
  time_at_end_table = 0
  trajectory_length = 0

  start_time = timestamps[0]
  last_robot_position = positions[0]

  for (timestamp, base_position) in zip(timestamps, positions):
    distance_moved = np.linalg.norm(last_robot_position - base_position)
    trajectory_length += distance_moved
    # Check if any of the obstacle gates are passed through.
    for key, gates_list in OBSTACLES_DICT.items():
      # Skip if the obstacle is already completed.
      if completed_gates[key] >= len(gates_list):
        continue
      active_gate = gates_list[completed_gates[key]]
      if _check_pass_through_gate(
          active_gate, base_position, last_robot_position):
        completed_gates[key] += 1
        # If it left the start table update the starting time.
        if key == "start table":
          start_time = timestamp

    if _robot_at_end_table(base_position):
      if time_reached_end_table == 0:
        time_reached_end_table = timestamp - 1e-6
      time_at_end_table = timestamp - time_reached_end_table
      if time_at_end_table >= END_TABLE_COMPLETION_TIME:
        break
    elif time_at_end_table > 0 and _robot_fell_outside_end_table(base_position):
      break

    last_robot_position = base_position

  time_spent = timestamp - start_time
  (allotted_course_time, num_obstacles_completed, progress_per_obstacle) = (
      _calculate_allotted_course_time(
          completed_gates, time_at_end_table, touched_jump
      )
  )
  agility_score = _calculate_agility_score(
      allotted_course_time,
      time_spent,
      num_obstacles_completed,
      time_at_end_table,
  )

  num_gates_completed = sum(completed_gates.values())
  total_gates = sum([len(x) for x in OBSTACLES_DICT.values()])
  course_time = time_spent - time_at_end_table
  metrics = {
      "gates_completed": num_gates_completed,
      "total_gates": total_gates,
      "progress_per_obstacle": progress_per_obstacle,
      "obstacles_completed": num_obstacles_completed,
      "time_at_end_table": time_at_end_table,
      "allotted_course_time_seconds": allotted_course_time,
      "final_run_time_seconds": course_time,
      "excess_time_seconds": course_time - allotted_course_time,
      "trajectory_length_meter": trajectory_length,
      "agility_score": agility_score
  }
  return agility_score, metrics


def _calculate_agility_score(
    allotted_course_time, time_spent, num_obstacles_completed, time_at_end_table
):
  penalty = PENALTY_PER_OBSTACLE * (
      len(OBSTACLES_DICT) - num_obstacles_completed)
  if time_at_end_table >= END_TABLE_COMPLETION_TIME:
    time_spent -= time_at_end_table
  else:
    time_spent = MAX_TIME + allotted_course_time
  return 1.0 - max(time_spent - allotted_course_time, 0) * 0.01 - penalty


def _calculate_allotted_course_time(
    completed_gates, time_at_end_table, touched_jump
):
  allotted_course_time = 0
  progress_per_obstacle = {}
  num_obstacles_completed = 0
  for obstacle, gates in OBSTACLES_DICT.items():
    progress_per_obstacle[obstacle] = completed_gates[
        obstacle] / len(gates)
    if obstacle == "jump" and touched_jump:
      progress_per_obstacle[obstacle] = 0
    if (obstacle == "end table" and time_at_end_table >= 5) or (
        obstacle != "end table" and
        progress_per_obstacle[obstacle] >= 1.0):
      num_obstacles_completed += 1
      allotted_course_time += ALLOTED_TIMES[obstacle]
  return allotted_course_time, num_obstacles_completed, progress_per_obstacle


def _robot_at_end_table(base_position):
  return max(np.abs(END_TABLE_POSITION - base_position)) < END_TABLE_HALF_WIDTH


def _robot_fell_outside_end_table(base_position):
  """Check if the robot is 110% outside the final table."""
  return (
      max(np.abs(END_TABLE_POSITION - base_position))
      > END_TABLE_HALF_WIDTH * 1.1
  )


def _check_pass_through_gate(gate, current_robot_position, last_robot_position):
  """Checks if the robot has passed through the gate or close enough to it."""
  distance_covered_since_last_step = np.linalg.norm(
      current_robot_position - last_robot_position
  )
  # Only do this check if the position jumped far enough to skip over the gate.
  if distance_covered_since_last_step >= 2 * SUCCESS_DISTANCE:
    return _check_2d_line_intersection(
        last_robot_position,
        current_robot_position,
        gate.left_pole,
        gate.right_pole,
    )
  # Otherwise we check the current distance to the gate line segment.
  gate_distance = _calculate_distance_to_line_segment(
      *current_robot_position, *gate.left_pole, *gate.right_pole)
  return gate_distance < SUCCESS_DISTANCE


def _calculate_distance_to_line_segment(x, y, x1, y1, x2, y2):
  """Calculates the shortest distance between a point and a line segment in 2D.

  Args:
    x: Point's x coordinate.
    y: Point's y coordinate.
    x1: Start of the line's x coordinate.
    y1: Start of the line's y coordinate.
    x2: End of the line's x coordinate.
    y2: End of the line's y coordinate.

  Returns:
    Shortest distance between the point and the line segment.
  """
  dist_x = x - x1
  dist_y = y - y1
  segment_x = x2 - x1
  segment_y = y2 - y1

  dot = dist_x * segment_x + dist_y * segment_y
  len_sq = segment_x * segment_x + segment_y * segment_y
  param = -1
  # Find the closest point in the segment.
  if len_sq > 0:
    param = dot / len_sq
  if param < 0:
    # The start of the segment is the closest point.
    point_x = x1
    point_y = y1
  elif param > 1:
    # The end of the segment is the closest point.
    point_x = x2
    point_y = y2
  else:
    # Linear interpolation to the closest point.
    point_x = x1 + param * segment_x
    point_y = y1 + param * segment_y

  return math.sqrt((x - point_x) * (x - point_x) + (y - point_y) *
                   (y - point_y))


def _orientation(p, q, r):
  """Find the orientation of an ordered triplet (p,q,r)."""
  orientation = (float(q[1] - p[1]) * (r[0] - q[0])) - (
      float(q[0] - p[0]) * (r[1] - q[1]))
  return np.sign(orientation)


def _check_2d_line_intersection(p1, q1, p2, q2):
  """Check if 2 lines intersect in 2D while ignoring colinear cases."""
  # Find the 4 orientations required.
  ori_1 = _orientation(p1, q1, p2)
  ori_2 = _orientation(p1, q1, q2)
  ori_3 = _orientation(p2, q2, p1)
  ori_4 = _orientation(p2, q2, q1)

  # General case.
  if ((ori_1 != ori_2) and (ori_3 != ori_4)):
    return True
  # We ignore colinear cases for the purpose of the barkour, because we check
  # the current distance of the robot to the gate.
  return False


def main(_):
  data = np.loadtxt(log_file_flag.value, delimiter=",")
  timestamps = data[:, 0]
  positions = data[:, 1:]
  score, metrics = score_barkour(
      timestamps, positions, touched_jumpboard_flag.value
  )
  print("Detailed metrics:")
  pprint.pprint(metrics)
  print(f"\nBarkour score: {score}\n")


if __name__ == "__main__":
  app.run(main)
