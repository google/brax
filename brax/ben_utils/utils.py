import brax
from sys import stdout
import numpy as np

def make_config(n_players=2, torque=False, walls=False, output_path=False):
  pitm = brax.Config(dt=0.05, substeps=20, dynamics_mode='pbd')
  ground = pitm.bodies.add(name='ground')
  ground.frozen.all = True
  plane = ground.colliders.add().plane
  plane.SetInParent()  # for setting an empty oneof

  # make ball
  ball = pitm.bodies.add(name='ball', mass=1)
  # cap = ball.colliders.add().capsule
  # cap.radius, cap.length = 0.5, 1.0
  cap = ball.colliders.add().sphere
  cap.radius = 0.5


  # make piggy
  piggy = pitm.bodies.add(name='piggy', mass=1)
  box = piggy.colliders.add().box
  dims = box.halfsize
  dims.x, dims.y, dims.z = .5, .5, .5

  # make players
  for i in range(1,n_players+1):
    player = pitm.bodies.add(name='p%d'%i, mass=1)
    cap = player.colliders.add().capsule
    cap.radius, cap.length = 0.5, 1
    # players' actuators
    actuators = []
    if torque:
      actuators = ['roll', 'pitch']
      for a in actuators:
        body_str = 'p%d_%s'%(i,a)
        joint_str = 'p%d_joint_%s'%(i,a)
        act_str = 'p%d_torque_%s'%(i,a)
        pitm.bodies.add(name=body_str, mass=1e-4)
        joint = pitm.joints.add(name=joint_str, 
                                parent=player.name, child=body_str,
                                # angular_damping=0,
                                )
        joint.angle_limit.add(min = -180, max = 180)
        if a is 'pitch': joint.rotation.z = -90
        act = pitm.actuators.add(name=act_str, joint=joint_str,
                                          strength=100).torque
        act.SetInParent()  # for setting an empty oneof


  if walls:
    for i in range(1,5):
      wall = pitm.bodies.add(name='wall%d'%i)
      wall.frozen.all = True
      wall_box = wall.colliders.add().box
      dims = wall_box.halfsize
      if i<3:
        dims.x, dims.y, dims.z = 15, 0.25, 3
      else:
        dims.x, dims.y, dims.z = 0.25, 15, 3
    
    
  pitm.gravity.z = -9.8
  pitm.friction = 10.
  pitm.elasticity = 1.
  pitm.angular_damping = -1.0

  sys = brax.System(pitm)

  # default starting positions
  default_qp = sys.default_qp()
  r = 4. # starting distance of each player from ball
  t = np.linspace(0, 2*np.pi, n_players+1)
  dx, dy = r*np.cos(t), r*np.sin(t)
  default_qp.pos[2] += np.array([8., 0., 0.]) # default piggy
  bodies_per_player = 1+len(actuators)*torque
  for i in range(n_players):
    for j in range(bodies_per_player): # 3 if torque, 1 if not
      default_qp.pos[3+bodies_per_player*i+j] += np.array([dx[i], dy[i], 0.])
  
  if walls:
    default_qp.pos[-1] += np.array([15, 0, 0])
    default_qp.pos[-2] += np.array([-15, 0, 0])
    default_qp.pos[-3] += np.array([0, 15, 0])
    # default_qp.rot[-3] += np.array([1, 0, 0, 1])
    default_qp.pos[-4] += np.array([0, -15, 0])
    # default_qp.rot[-4] += np.array([1, 0, 0, 1])

    if output_path:
        original_stdout = stdout # Save a reference to the original standard output
        with open(join(output_path, 'config.py'), 'w') as f:
            stdout = f
            print("\"\"\"\n", env.sys.config, "\n\"\"\"")
            stdout = original_stdout # Reset the standard output to its original value

  return pitm, sys, default_qp