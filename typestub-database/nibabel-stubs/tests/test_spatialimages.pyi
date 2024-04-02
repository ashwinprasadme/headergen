from _typeshed import Incomplete

from ..imageclasses import spatial_axes_first as spatial_axes_first
from ..spatialimages import HeaderDataError as HeaderDataError
from ..spatialimages import SpatialHeader as SpatialHeader
from ..spatialimages import SpatialImage as SpatialImage
from ..testing import bytesio_round_trip as bytesio_round_trip
from ..testing import deprecated_to as deprecated_to
from ..testing import expires as expires
from ..testing import memmap_after_ufunc as memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory as InTemporaryDirectory

def test_header_init() -> None: ...
def test_from_header(): ...
def test_eq() -> None: ...
def test_copy() -> None: ...
def test_shape_zooms() -> None: ...
def test_data_dtype() -> None: ...
def test_affine() -> None: ...
def test_read_data() -> None: ...

class DataLike:
    shape: Incomplete
    def __array__(self, dtype: str = "int16"): ...

class TestSpatialImage:
    image_class = SpatialImage
    can_save: bool
    def test_isolation(self) -> None: ...
    def test_float_affine(self) -> None: ...
    def test_images(self) -> None: ...
    def test_default_header(self) -> None: ...
    def test_data_api(self) -> None: ...
    def check_dtypes(self, expected, actual) -> None: ...
    def test_data_default(self) -> None: ...
    def test_data_shape(self) -> None: ...
    def test_str(self) -> None: ...
    def test_get_fdata(self) -> None: ...
    def test_get_data(self) -> None: ...
    def test_slicer(self) -> None: ...

class MmapImageMixin:
    check_mmap_mode: bool
    def get_disk_image(self): ...
    def test_load_mmap(self) -> None: ...
