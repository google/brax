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

"""Tests for parameter saving/loading."""

import os
import pickle
import tempfile

from absl.testing import absltest
import jax
import jax.numpy as jnp

from brax.io import model as brax_model


class ModelTest(absltest.TestCase):

  def test_save_load_params(self):
    """Verifies that Msgpack serialization preserves Pytree data integrity."""
    params = {
        'policy': {
            'w': jnp.ones((4, 8)),
            'b': jnp.zeros((8,)),
        },
        'stats': (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])),
        'list': [jnp.array(1), jnp.array(2)],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'params.msgpack')
      brax_model.save_params(path, params)
      loaded_params = brax_model.load_params(path)

    # Check structure and values
    import numpy as np

    jax.tree_util.tree_map(np.testing.assert_allclose, params, loaded_params)

  def test_pickle_security_block(self):
    """Verifies that legacy pickle files are blocked by default."""
    params = {'test': 123}
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'params.pkl')
      with open(path, 'wb') as f:
        f.write(pickle.dumps(params))

      with self.assertRaisesRegex(
          RuntimeError, 'SECURITY ERROR: Insecure pickle file'
      ):
        brax_model.load_params(path)

  def test_pickle_allow_explicit(self):
    """Verifies that legacy files can still be loaded with explicit flag."""
    params = {'test': 456}
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'params.pkl')
      with open(path, 'wb') as f:
        f.write(pickle.dumps(params))

      with self.assertWarns(brax_model.SecurityWarning):
        loaded_params = brax_model.load_params(path, allow_pickle=True)

    self.assertEqual(params, loaded_params)

  def test_rce_prevention(self):
    """Verifies that malicious payloads are blocked before deserialization."""

    class Malicious:

      def __reduce__(self):
        return (os.system, ('echo RCE_EXPLOITED',))

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'malicious.pkl')
      with open(path, 'wb') as f:
        f.write(pickle.dumps(Malicious()))

      # Should raise RuntimeError and NOT execute the payload
      with self.assertRaises(RuntimeError):
        brax_model.load_params(path)


if __name__ == '__main__':
  absltest.main()
