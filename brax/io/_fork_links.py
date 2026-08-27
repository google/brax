# explore_bench fork of brax: link/body decoupling for the mjcf loader.
"""Builds brax links from a MuJoCo model without assuming link == body.

Stock brax emits ONE LINK PER JOINT GROUP while indexing every per-link array
BY BODY (``link.transform`` is ``mj.body_pos``, ``link_parents`` is
``mj.body_parentid``, and ``contact`` resolves a geom's link as
``geom_bodyid - 1``).  Those two indexings agree only when every body carries
exactly one joint.  Two common MJCF constructions break the correspondence:

* a **welded body** -- a body with no joint at all, rigidly attached to its
  parent.  It contributes a row to the per-body arrays and no entry to the
  per-joint-group ones, so the arrays desynchronise.  Stock brax does not
  error; it builds a System whose ``link.joint`` is shorter than
  ``link.transform``, and the mismatch surfaces much later as a ``vmap got
  inconsistent sizes`` deep inside ``kinematics.forward``.
* a **joint stack wider than 3 DOF** -- brax's positional pipeline resolves a
  joint as at most a spherical (3-DOF) rotation plus translation, so a body
  carrying, say, three slides and two hinges has no single-link representation
  at all.

This module builds an explicit body -> link map that handles both:

* welded bodies are FUSED into their nearest jointed ancestor (mass and
  inertia merged by the parallel-axis theorem, then re-diagonalised onto
  principal axes because ``com.inv_inertia`` reads only ``diagonal(i)``);
  a body welded to the world becomes static world geometry (link -1);
* a stack of n > 3 DOFs is SPLIT across ``ceil(n / 3)`` chained links joined
  by massless intermediates, the last of which carries the body's inertia and
  geometry.

Both transformations are exact for rigid-body dynamics and neither touches
``nq``/``nv``, the joint order, or the dof ordering, so ``qpos``/``qvel``
transfer between the MuJoCo model and the brax System stays a straight copy.
"""

import numpy as np

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

#: Mass (and inertia) share handed to the massless intermediate link of a
#: split joint stack, as a fraction of its host body. Small enough to be a
#: minor perturbation, large enough that 1/mass stays well conditioned next to
#: the other links in these scenes.
SPLIT_MASS_FRACTION = 1e-2


def _quat_mul(u, v):
  """Hamilton product, wxyz (matches brax.math.quat_mul)."""
  w0, x0, y0, z0 = u
  w1, x1, y1, z1 = v
  return np.array([
      w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
      w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
      w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
  ])


def _quat_to_mat(q):
  w, x, y, z = q
  return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ])


def _mat_to_quat(m):
  """Rotation matrix -> wxyz quaternion (Shepperd's method, branch on trace)."""
  tr = m[0, 0] + m[1, 1] + m[2, 2]
  if tr > 0:
    s = np.sqrt(tr + 1.0) * 2
    q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                  (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
  elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
    s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
    q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                  (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
  elif m[1, 1] > m[2, 2]:
    s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
    q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                  0.25 * s, (m[1, 2] + m[2, 1]) / s])
  else:
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                  (m[1, 2] + m[2, 1]) / s, 0.25 * s])
  n = np.linalg.norm(q)
  return q / n if n > 0 else _IDENTITY_QUAT.copy()


def _rotate(v, q):
  return _quat_to_mat(q) @ v


def _compose(pos_a, quat_a, pos_b, quat_b):
  """Frame b expressed in a's parent frame, given b relative to a."""
  return pos_a + _rotate(pos_b, quat_a), _quat_mul(quat_a, quat_b)


class LinkMap:
  """Body/geom -> link mapping plus the per-link arrays brax needs."""

  def __init__(self, mj):
    self.mj = mj
    self._build()

  # -- helpers ------------------------------------------------------------
  def _rel_transform(self, body, stop_body):
    """Pose of ``body`` expressed in ``stop_body``'s frame."""
    mj = self.mj
    chain = []
    cur = int(body)
    while cur != int(stop_body) and cur > 0:
      chain.append(cur)
      cur = int(mj.body_parentid[cur])
    pos = np.zeros(3)
    quat = _IDENTITY_QUAT.copy()
    for c in reversed(chain):
      pos, quat = _compose(pos, quat, mj.body_pos[c], mj.body_quat[c])
    return pos, quat

  # -- construction -------------------------------------------------------
  def _build(self):
    mj = self.mj
    n_dofs_of_body = {}
    first_jnt_of_body = {}
    for j in range(mj.njnt):
      b = int(mj.jnt_bodyid[j])
      typ = int(mj.jnt_type[j])
      n_dofs_of_body[b] = n_dofs_of_body.get(b, 0) + (6 if typ == 0 else
                                                      3 if typ == 1 else 1)
      first_jnt_of_body.setdefault(b, j)
    self.n_dofs_of_body = n_dofs_of_body

    # link_bodies[i] is the body that link i belongs to; a body split across
    # several links appears once per link, and link_of_body points at the LAST
    # (the one carrying the body's mass and geometry).
    link_bodies, link_types, link_parents, link_joint_pos = [], [], [], []
    link_names, link_is_terminal = [], []
    link_of_body = np.full(mj.nbody, -1, dtype=np.int64)
    self.split_bodies = {}

    for b in range(1, mj.nbody):
      parent = int(mj.body_parentid[b])
      if b not in n_dofs_of_body:                    # welded -> fuse
        link_of_body[b] = link_of_body[parent]
        continue
      n_dof = n_dofs_of_body[b]
      j0 = first_jnt_of_body[b]
      if n_dof == 6 and int(mj.jnt_type[j0]) == 0:
        groups = [('f', 6)]
      else:
        groups, left = [], n_dof
        while left > 0:
          take = min(3, left)
          groups.append((str(take), take))
          left -= take
      if len(groups) > 1:
        self.split_bodies[b] = len(groups)
      parent_link = link_of_body[parent]
      base_name = _body_name(mj, b)
      for k, (typ, _) in enumerate(groups):
        idx = len(link_bodies)
        link_bodies.append(b)
        link_types.append(typ)
        link_parents.append(int(parent_link))
        link_joint_pos.append(np.array(mj.jnt_pos[j0]))
        link_names.append(base_name if k == len(groups) - 1
                          else f'{base_name}#split{k}')
        link_is_terminal.append(k == len(groups) - 1)
        parent_link = idx
      link_of_body[b] = parent_link                  # last link of the chain

    self.link_bodies = np.array(link_bodies, dtype=np.int64)
    self.link_types = ''.join(link_types)
    self.link_parents = tuple(int(p) for p in link_parents)
    self.link_names = link_names
    self.link_joint_pos = np.array(link_joint_pos) if link_joint_pos else \
        np.zeros((0, 3))
    self.link_is_terminal = np.array(link_is_terminal, dtype=bool)
    self.link_of_body = link_of_body
    self.num_links = len(link_bodies)

    self._build_transforms()
    self._build_inertia()
    self._build_geoms()

  def _build_transforms(self):
    """Parent-link-relative rest transform for every link."""
    mj = self.mj
    pos = np.zeros((self.num_links, 3))
    quat = np.tile(_IDENTITY_QUAT, (self.num_links, 1))
    for i in range(self.num_links):
      if not self.link_is_terminal[i]:
        continue                                     # massless intermediate
      b = int(self.link_bodies[i])
      p_link = self.link_parents[i]
      stop = int(self.link_bodies[p_link]) if p_link >= 0 else 0
      # a split body's chain sits between its parent link and itself; the
      # intermediates are identity, so the whole body offset rides the last
      p, q = self._rel_transform(b, stop)
      pos[i], quat[i] = p, q
    self.link_pos, self.link_quat = pos, quat

  def _build_inertia(self):
    """Merge welded descendants into the link that carries them."""
    mj = self.mj
    mass = np.zeros(self.num_links)
    ipos = np.zeros((self.num_links, 3))
    iquat = np.tile(_IDENTITY_QUAT, (self.num_links, 1))
    idiag = np.zeros((self.num_links, 3))
    invweight = np.zeros(self.num_links)

    members = {i: [] for i in range(self.num_links)}
    for b in range(1, mj.nbody):
      li = int(self.link_of_body[b])
      if li >= 0:
        members[li].append(b)

    for i in range(self.num_links):
      if not self.link_is_terminal[i]:
        # A split introduces a virtual frame that carries no mass in
        # generalized coordinates. brax is a MAXIMAL-coordinate solver: it
        # forms 1/mass and 1/diag(i) per link, so a massless link is a
        # division by zero (measured: every DOF of the split body goes NaN on
        # the first step). Give the intermediate a small share of its host's
        # mass and inertia -- it sits at the host's frame, so the share
        # translates with the body exactly as the host's own mass does. The
        # host keeps its FULL inertia, so rotation about the split joint is
        # unchanged; the cost is SPLIT_MASS_FRACTION extra translating mass.
        host = int(self.link_bodies[i])
        mass[i] = SPLIT_MASS_FRACTION * float(mj.body_mass[host])
        idiag[i] = SPLIT_MASS_FRACTION * np.asarray(mj.body_inertia[host],
                                                    float)
        invweight[i] = float(mj.body_invweight0[host, 0])
        continue
      host = int(self.link_bodies[i])
      invweight[i] = float(mj.body_invweight0[host, 0])
      bodies = members[i]
      if len(bodies) == 1 and bodies[0] == host:     # fast path: untouched
        mass[i] = float(mj.body_mass[host])
        ipos[i] = mj.body_ipos[host]
        iquat[i] = mj.body_iquat[host]
        idiag[i] = mj.body_inertia[host]
        continue
      # merge every member's inertia into the host frame
      total_m, com = 0.0, np.zeros(3)
      terms = []
      for b in bodies:
        m_b = float(mj.body_mass[b])
        p, q = self._rel_transform(b, host)
        c_b = p + _rotate(mj.body_ipos[b], q)
        r_b = _quat_mul(q, mj.body_iquat[b])
        terms.append((m_b, c_b, r_b, np.asarray(mj.body_inertia[b], float)))
        total_m += m_b
        com += m_b * c_b
      com = com / total_m if total_m > 0 else com
      inertia = np.zeros((3, 3))
      for m_b, c_b, r_b, diag_b in terms:
        rot = _quat_to_mat(r_b)
        i_b = rot @ np.diag(diag_b) @ rot.T          # into host axes
        d = c_b - com
        inertia += i_b + m_b * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
      # com.inv_inertia reads only diagonal(i), so express on principal axes
      evals, evecs = np.linalg.eigh(inertia)
      if np.linalg.det(evecs) < 0:
        evecs[:, 0] = -evecs[:, 0]
      mass[i] = total_m
      ipos[i] = com
      iquat[i] = _mat_to_quat(evecs)
      idiag[i] = np.clip(evals, 1e-12, None)
    self.link_mass = mass
    self.link_ipos, self.link_iquat = ipos, iquat
    self.link_idiag = idiag
    self.link_invweight = invweight

  def _build_geoms(self):
    """Each geom's link, and its pose in that link's frame."""
    mj = self.mj
    n = mj.ngeom
    idx = np.zeros(n, dtype=np.int64)
    pos = np.zeros((n, 3))
    quat = np.tile(_IDENTITY_QUAT, (n, 1))
    for g in range(n):
      gb = int(mj.geom_bodyid[g])
      li = int(self.link_of_body[gb]) if gb > 0 else -1
      idx[g] = li
      stop = int(self.link_bodies[li]) if li >= 0 else 0
      p, q = self._rel_transform(gb, stop)
      pos[g], quat[g] = _compose(p, q, mj.geom_pos[g], mj.geom_quat[g])
    self.geom_link_idx = idx
    self.geom_link_pos = pos
    self.geom_link_quat = quat


def _body_name(mj, i):
  import mujoco
  name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(i))
  return name if name else f'body{i}'
