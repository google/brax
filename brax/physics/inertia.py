# Copyright 2022 The Brax Authors.
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

"""Functions to calculate and infer inertias."""

from brax import math
from brax import jumpy as jp
from brax.physics import config_pb2
from brax.physics.base import vec_to_arr
from google.protobuf import text_format


def quaternion_rotation_matrix(Q: jp.ndarray):
  """
  Covert a quaternion into a full three-dimensional rotation matrix.
 
  Input
  :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
  Output
  :return: A 3x3 element matrix representing the full 3D rotation matrix. 
       This rotation matrix converts a point in the local reference 
       frame to a point in the global reference frame.
  """
  # Extract the values from Q
  q0 = Q[0]
  q1 = Q[1]
  q2 = Q[2]
  q3 = Q[3]
   
  # First row of the rotation matrix
  r00 = 2 * (q0 * q0 + q1 * q1) - 1
  r01 = 2 * (q1 * q2 - q0 * q3)
  r02 = 2 * (q1 * q3 + q0 * q2)
   
  # Second row of the rotation matrix
  r10 = 2 * (q1 * q2 + q0 * q3)
  r11 = 2 * (q0 * q0 + q2 * q2) - 1
  r12 = 2 * (q2 * q3 - q0 * q1)
   
  # Third row of the rotation matrix
  r20 = 2 * (q1 * q3 - q0 * q2)
  r21 = 2 * (q2 * q3 + q0 * q1)
  r22 = 2 * (q0 * q0 + q3 * q3) - 1
   
  # 3x3 rotation matrix
  rot_matrix = jp.array([[r00, r01, r02],
               [r10, r11, r12],
               [r20, r21, r22]])
              
  return rot_matrix


def inertia_off_center(mass: float, vec: jp.ndarray):
  """Off-center correction inertia tensor (https://en.wikipedia.org/wiki/Parallel_axis_theorem)"""
  R = jp.sum(vec ** 2)
  I = mass * (jp.diag([R, R, R]) - jp.outer(vec, vec))
  return I


def inertia_rotated(inertia_diag: jp.ndarray, rot_euler_deg: jp.ndarray):
  """Rotate inertia tensor (https://hepweb.ucsd.edu/ph110b/110b_notes/node24.html)"""
  M = quaternion_rotation_matrix(math.euler_to_quat(rot_euler_deg))
  return M @ jp.diag(inertia_diag) @ M.transpose()


def inertia_from_geometry(collider: config_pb2.Collider, density: float = 1000.0):
  """Returns original inertia of geometry (without rotation and position)

  Reference: SetInertia in https://github.com/deepmind/mujoco/blob/main/src/user/user_objects.cc"""

  geom_type = collider.WhichOneof('type')

  if geom_type == "capsule":
    # Capsule inertia
    R = collider.capsule.radius
    H = collider.capsule.length - 2 * collider.capsule.radius
    # mass
    M_cylinder   = density * H * (R ** 2) * jp.pi
    M_hemisphere = density * 2 * (R ** 3) * jp.pi / 3
    M = M_cylinder + 2 * M_hemisphere

    # inertia
    Ixx = Iyy = M_cylinder * (((H ** 2) / 12) + ((R ** 2) / 4)) + \
          2 * M_hemisphere * ((2 * (R ** 2) / 5) + ((H ** 2) / 4) + (3 * H * R / 8))
    Izz =     M_cylinder * ((R ** 2) / 2) + \
          2 * M_hemisphere * (2 * (R ** 2) / 5)

    return M, jp.array([Ixx, Iyy, Izz])

  raise NotImplementedError("Inertia calculation of {} is not implemented!".format(geom_type))


def infer_inertia(config: config_pb2.Config, density: float = 1000):
  for body in config.bodies:
    # check if inferring is needed
    infer_mass = body.mass == 0
    infer_inertia = body.inertia.x == 0 and body.inertia.y == 0 and body.inertia.z == 0
    if not (infer_mass or infer_inertia):
      continue

    # ignore frozen bodies
    if body.frozen.all:
      # set arbitary nonzero mass for frozen bodies
      mass = 1.
      inertia = jp.diag([1., 1., 1.])
    else:
      # calculate inertia from colliders
      mass = 0
      inertia = jp.zeros((3, 3))

      for collider in body.colliders:
        geom_mass, geom_inertia_diag = inertia_from_geometry(collider, density)

        mass += geom_mass
        inertia += inertia_rotated(geom_inertia_diag, vec_to_arr(collider.rotation))
        inertia += inertia_off_center(geom_mass,    vec_to_arr(collider.position))

    # FIXME: Physics only use diagonal of inertia, approximate it here
    inertia = jp.diag(inertia)
  
    if infer_mass:
      body.mass = mass
    if infer_inertia:
      body.inertia.x = inertia[0]
      body.inertia.y = inertia[1]
      body.inertia.z = inertia[2]

  return config
