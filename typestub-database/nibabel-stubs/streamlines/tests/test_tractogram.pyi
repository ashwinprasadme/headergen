import unittest

from _typeshed import Incomplete

from ...testing import assert_arrays_equal as assert_arrays_equal
from ...testing import clear_and_catch_warnings as clear_and_catch_warnings
from ..tractogram import LazyDict as LazyDict
from ..tractogram import LazyTractogram as LazyTractogram
from ..tractogram import PerArrayDict as PerArrayDict
from ..tractogram import PerArraySequenceDict as PerArraySequenceDict
from ..tractogram import Tractogram as Tractogram
from ..tractogram import TractogramItem as TractogramItem
from ..tractogram import is_data_dict as is_data_dict
from ..tractogram import is_lazy_dict as is_lazy_dict

DATA: Incomplete

def make_fake_streamline(
    nb_points,
    data_per_point_shapes={},
    data_for_streamline_shapes={},
    rng: Incomplete | None = None,
): ...
def make_fake_tractogram(
    list_nb_points,
    data_per_point_shapes={},
    data_for_streamline_shapes={},
    rng: Incomplete | None = None,
): ...
def make_dummy_streamline(nb_points): ...
def setup_module(): ...
def check_tractogram_item(
    tractogram_item, streamline, data_for_streamline={}, data_for_points={}
) -> None: ...
def assert_tractogram_item_equal(t1, t2) -> None: ...
def check_tractogram(
    tractogram, streamlines=[], data_per_streamline={}, data_per_point={}
) -> None: ...
def assert_tractogram_equal(t1, t2) -> None: ...
def extender(a, b): ...

class TestPerArrayDict(unittest.TestCase):
    def test_per_array_dict_creation(self) -> None: ...
    def test_getitem(self) -> None: ...
    def test_extend(self) -> None: ...

class TestPerArraySequenceDict(unittest.TestCase):
    def test_per_array_sequence_dict_creation(self) -> None: ...
    def test_getitem(self) -> None: ...
    def test_extend(self) -> None: ...

class TestLazyDict(unittest.TestCase):
    def test_lazydict_creation(self) -> None: ...

class TestTractogramItem(unittest.TestCase):
    def test_creating_tractogram_item(self) -> None: ...

class TestTractogram(unittest.TestCase):
    def test_tractogram_creation(self) -> None: ...
    def test_setting_affine_to_rasmm(self) -> None: ...
    def test_tractogram_getitem(self) -> None: ...
    def test_tractogram_add_new_data(self) -> None: ...
    def test_tractogram_copy(self) -> None: ...
    def test_creating_invalid_tractogram(self) -> None: ...
    def test_tractogram_apply_affine(self) -> None: ...
    def test_tractogram_to_world(self) -> None: ...
    def test_tractogram_extend(self) -> None: ...

class TestLazyTractogram(unittest.TestCase):
    def test_lazy_tractogram_creation(self) -> None: ...
    def test_lazy_tractogram_from_data_func(self): ...
    def test_lazy_tractogram_getitem(self) -> None: ...
    def test_lazy_tractogram_extend(self) -> None: ...
    def test_lazy_tractogram_len(self) -> None: ...
    def test_lazy_tractogram_apply_affine(self) -> None: ...
    def test_tractogram_to_world(self) -> None: ...
    def test_lazy_tractogram_copy(self) -> None: ...