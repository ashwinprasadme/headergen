from _typeshed import Incomplete
from nibabel.tmpdirs import InTemporaryDirectory as InTemporaryDirectory
from numpy.testing import assert_array_almost_equal as assert_array_almost_equal

from ... import load as load
from ...fileholders import FileHolder as FileHolder
from ...nifti1 import data_type_codes as data_type_codes
from ...testing import get_test_data as get_test_data
from .. import GiftiCoordSystem as GiftiCoordSystem
from .. import GiftiDataArray as GiftiDataArray
from .. import GiftiImage as GiftiImage
from .. import GiftiLabel as GiftiLabel
from .. import GiftiLabelTable as GiftiLabelTable
from .. import GiftiMetaData as GiftiMetaData
from .. import GiftiNVPairs as GiftiNVPairs
from .test_parse_gifti_fast import DATA_FILE1 as DATA_FILE1
from .test_parse_gifti_fast import DATA_FILE2 as DATA_FILE2
from .test_parse_gifti_fast import DATA_FILE3 as DATA_FILE3
from .test_parse_gifti_fast import DATA_FILE4 as DATA_FILE4
from .test_parse_gifti_fast import DATA_FILE5 as DATA_FILE5
from .test_parse_gifti_fast import DATA_FILE6 as DATA_FILE6

rng: Incomplete

def test_agg_data() -> None: ...
def test_gifti_image() -> None: ...
def test_gifti_image_bad_inputs() -> None: ...
def test_image_typing(label) -> None: ...
def test_dataarray_empty() -> None: ...
def test_dataarray_init() -> None: ...
def test_dataarray_typing(label) -> None: ...
def test_labeltable() -> None: ...
def test_metadata() -> None: ...
def test_metadata_list_interface() -> None: ...
def test_gifti_label_rgba() -> None: ...
def test_print_summary(fname, capsys) -> None: ...
def test_gifti_coord(capsys) -> None: ...
def test_gifti_round_trip() -> None: ...
def test_data_array_round_trip() -> None: ...
def test_darray_dtype_coercion_failures() -> None: ...
def test_gifti_file_close(recwarn) -> None: ...
