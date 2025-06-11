# Copyright 2025 The Brax Authors.
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

"""Tests for brax training types."""

from absl.testing import absltest
from brax.training import types
import jax
from jax import numpy as jp

UInt64 = types.UInt64


class TestUInt64(absltest.TestCase):
  """Tests for UInt64 class and its operations."""

  def test_add_uint64_basic(self):
    """Tests basic addition of two UInt64 numbers."""
    a = UInt64(hi=jp.array(0), lo=jp.array(1))
    b = UInt64(hi=jp.array(0), lo=jp.array(2))
    expected = UInt64(hi=jp.array(0), lo=jp.array(3))
    self.assertEqual(a + b, expected)

  def test_add_uint64_with_carry(self):
    """Tests addition of two UInt64 numbers with carry-over."""
    max_uint32 = 2**32 - 1
    a = UInt64(hi=jp.array(0), lo=jp.array(max_uint32, dtype=jp.uint32))
    b = UInt64(hi=jp.array(0), lo=jp.array(1))
    expected = UInt64(hi=jp.array(1), lo=jp.array(0))
    self.assertEqual(a + b, expected)

  def test_add_uint64_large_numbers(self):
    """Tests addition with larger numbers that will result in non-zero hi."""
    a = UInt64(hi=jp.array(1), lo=jp.array(0))
    b = UInt64(hi=jp.array(2), lo=jp.array(0))
    expected = UInt64(hi=jp.array(3), lo=jp.array(0))
    self.assertEqual(a + b, expected)

  def test_add_uint64_max_values(self):
    """Tests addition of the maximum possible UInt64 values."""
    max_uint32 = jp.array(2**32 - 1, dtype=jp.uint32)
    a = UInt64(hi=max_uint32, lo=max_uint32)
    b = UInt64(hi=max_uint32, lo=max_uint32)
    expected_hi = jp.array(2**32 - 1, dtype=jp.uint32)
    expected_lo = jp.array(2**32 - 2, dtype=jp.uint32)
    expected = UInt64(hi=expected_hi, lo=expected_lo)
    self.assertEqual(a + b, expected)

  def test_uint64_add_different_type(self):
    """Tests adding UInt64 with a non-UInt64 type."""
    a = UInt64(hi=jp.array(0), lo=jp.array(1))
    self.assertEqual(a + 1, UInt64(hi=jp.array(0), lo=jp.array(2)))

  def test_uint64_repr(self):
    """Tests the __repr__ method of UInt64."""
    a = UInt64(hi=jp.array(1), lo=jp.array(20))
    self.assertEqual(repr(a), f"{UInt64(hi=jp.array(1), lo=jp.array(20))}")

  def test_add_uint64_jit(self):
    """Tests UInt64 addition under jit."""

    @jax.jit
    def add_uint64(a, b):
      return a + b

    a = UInt64(hi=jp.array(1), lo=jp.array(2**32 - 1, dtype=jp.uint32))
    b = UInt64(hi=jp.array(2), lo=jp.array(1))
    result = add_uint64(a, b)
    expected = UInt64(hi=jp.array(4), lo=jp.array(0))
    self.assertEqual(result, expected)

  def test_uint64_to_numpy(self):
    """Tests the to_numpy method of UInt64."""
    a = UInt64(hi=jp.array(1), lo=jp.array(5120))
    self.assertEqual(a.to_numpy(), (1 << 32) + 5120)

  def test_uint64_int(self):
    """Tests the int method of UInt64."""
    a = UInt64(hi=jp.array(1), lo=jp.array(20))
    self.assertEqual(int(a), (1 << 32) + 20)


if __name__ == "__main__":
  absltest.main()
