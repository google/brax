# Copyright 2026 The Brax Authors.
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

"""Tests that environment docstrings match their MJCF XML definitions."""

import ast
import pathlib
import re
from xml.etree import ElementTree

from absl.testing import absltest

_ENVS_DIR = pathlib.Path(__file__).resolve().parent.parent / 'brax' / 'envs'
_ASSETS_DIR = _ENVS_DIR / 'assets'


def _get_class_docstring(env_file: str, class_name: str) -> str:
  tree = ast.parse((_ENVS_DIR / env_file).read_text(encoding='utf-8'))
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == class_name:
      doc = ast.get_docstring(node, clean=False)
      assert doc is not None
      return doc
  raise AssertionError(f'Class {class_name} not found in {env_file}')


def _parse_markdown_table(section: str) -> list[list[str]]:
  rows = []
  for line in section.splitlines():
    stripped = line.strip()
    if not stripped.startswith('|'):
      continue
    cells = [cell.strip() for cell in stripped.strip('|').split('|')]
    if not cells or cells[0] == 'Num' or set(cells[0]) == {'-'}:
      continue
    rows.append(cells)
  return rows


def _section(doc: str, header: str) -> str:
  pattern = rf'### {re.escape(header)}\s*(.*?)(?=\n\s*### |\Z)'
  match = re.search(pattern, doc, flags=re.DOTALL)
  assert match is not None, f'Section {header} not found'
  return match.group(1)


def _xml_actuators(xml_file: str) -> list[str]:
  root = ElementTree.fromstring(
      (_ASSETS_DIR / xml_file).read_text(encoding='utf-8')
  )
  return [elem.attrib['joint'] for elem in root.findall('./actuator/motor')]


def _xml_joints(xml_file: str) -> list[str]:
  root = ElementTree.fromstring(
      (_ASSETS_DIR / xml_file).read_text(encoding='utf-8')
  )
  joints = []

  def walk(elem: ElementTree.Element) -> None:
    for child in elem:
      if child.tag == 'joint':
        joints.append(child.attrib['name'])
      else:
        walk(child)

  worldbody = root.find('worldbody')
  assert worldbody is not None
  walk(worldbody)
  return joints


def _xml_fused_link_count(xml_file: str) -> int:
  root = ElementTree.fromstring(
      (_ASSETS_DIR / xml_file).read_text(encoding='utf-8')
  )
  worldbody = root.find('worldbody')
  assert worldbody is not None
  return sum(
      1
      for body in worldbody.iter('body')
      if body.find('joint') is not None or body.find('freejoint') is not None
  )


class EnvsDocTest(absltest.TestCase):

  def test_ant_action_and_observation_ordering(self):
    doc = _get_class_docstring('ant.py', 'Ant')

    action_rows = _parse_markdown_table(_section(doc, 'Action Space'))
    expected_actuators = _xml_actuators('ant.xml')
    self.assertLen(action_rows, len(expected_actuators))
    for idx, (row, joint) in enumerate(zip(action_rows, expected_actuators)):
      self.assertEqual(row[0], str(idx))
      self.assertTrue(
          row[4].startswith(f'{joint} '),
          f'Action row {idx} expected {joint}, got {row[4]}',
      )

    obs_rows = _parse_markdown_table(_section(doc, 'Observation Space'))
    hinge_joints = _xml_joints('ant.xml')[1:]  # skip root free joint
    self.assertLen(obs_rows, 5 + len(hinge_joints) + 6 + len(hinge_joints))
    for offset, joint in enumerate(hinge_joints):
      self.assertTrue(
          obs_rows[5 + offset][4].startswith(f'{joint} '),
          f'Obs qpos row {5 + offset} expected {joint}, got '
          f'{obs_rows[5 + offset][4]}',
      )
      vel_idx = 5 + len(hinge_joints) + 6 + offset
      self.assertTrue(
          obs_rows[vel_idx][4].startswith(f'{joint} '),
          f'Obs qvel row {vel_idx} expected {joint}, got '
          f'{obs_rows[vel_idx][4]}',
      )

    term_section = _section(doc, 'Episode Termination')
    self.assertIn('z-coordinate of the torso (index 0)', term_section)

  def test_humanoid_and_standup_spaces_match_xml(self):
    for env_file, class_name, xml_file in [
        ('humanoid.py', 'Humanoid', 'humanoid.xml'),
        ('humanoidstandup.py', 'HumanoidStandup', 'humanoidstandup.xml'),
    ]:
      with self.subTest(env_file=env_file):
        doc = _get_class_docstring(env_file, class_name)

        action_rows = _parse_markdown_table(_section(doc, 'Action Space'))
        expected_actuators = _xml_actuators(xml_file)
        self.assertLen(action_rows, len(expected_actuators))
        for idx, (row, joint) in enumerate(
            zip(action_rows, expected_actuators)
        ):
          self.assertEqual(row[0], str(idx))
          self.assertTrue(
              row[4] == joint or row[4].startswith(f'{joint} '),
              f'{env_file} action row {idx} expected {joint}, got {row[4]}',
          )
          if joint.endswith(('_x', '_y', '_z')):
            self.assertIn(f'{joint[-1]}-coordinate', row[1])

        obs_section = _section(doc, 'Observation Space')
        nlinks = _xml_fused_link_count(xml_file)
        hinge_joints = _xml_joints(xml_file)[1:]  # skip root free joint
        nq = 5 + len(hinge_joints)
        nv = 6 + len(hinge_joints)
        expected_obs_dim = nq + nv + nlinks * 10 + nlinks * 6 + nv
        self.assertIn(f'shape `({expected_obs_dim},)`', obs_section)

        obs_rows = _parse_markdown_table(obs_section)
        self.assertLen(obs_rows, nq + nv)
        for offset, joint in enumerate(hinge_joints):
          qpos_row = obs_rows[5 + offset]
          qvel_row = obs_rows[nq + 6 + offset]
          self.assertEqual(qpos_row[4], joint)
          self.assertEqual(qvel_row[4], joint)
          if joint.endswith(('_x', '_y', '_z')):
            axis = joint[-1]
            self.assertTrue(
                f'{axis}-angle' in qpos_row[1]
                or f'{axis}-coordinate' in qpos_row[1]
            )
            self.assertIn(f'{axis}-coordinate', qvel_row[1])


if __name__ == '__main__':
  absltest.main()
