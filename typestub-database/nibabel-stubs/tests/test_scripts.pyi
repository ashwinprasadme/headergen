from _typeshed import Incomplete

from ..loadsave import load as load
from ..orientations import aff2axcodes as aff2axcodes
from ..orientations import inv_ornt_aff as inv_ornt_aff
from ..testing import assert_data_similar as assert_data_similar
from ..testing import assert_dt_equal as assert_dt_equal
from ..testing import assert_re_in as assert_re_in
from ..tmpdirs import InTemporaryDirectory as InTemporaryDirectory
from .nibabel_data import needs_nibabel_data as needs_nibabel_data
from .scriptrunner import ScriptRunner as ScriptRunner
from .test_parrec import DTI_PAR_BVALS as DTI_PAR_BVALS
from .test_parrec import DTI_PAR_BVECS as DTI_PAR_BVECS
from .test_parrec_data import AFF_OFF as AFF_OFF
from .test_parrec_data import BALLS as BALLS

runner: Incomplete
run_command: Incomplete

def script_test(func): ...

DATA_PATH: Incomplete

def load_small_file(): ...
def check_nib_ls_example4d(
    opts=[], hdrs_str: str = "", other_str: str = ""
) -> None: ...
def check_nib_diff_examples() -> None: ...
def test_nib_ls(args) -> None: ...
def test_nib_ls_multiple() -> None: ...
def test_help() -> None: ...
def test_nib_diff() -> None: ...
def test_nib_nifti_dx() -> None: ...
def vox_size(affine): ...
def check_conversion(cmd, pr_data, out_fname) -> None: ...
def test_parrec2nii() -> None: ...
def test_parrec2nii_with_data() -> None: ...
def test_nib_trk2tck() -> None: ...
def test_nib_tck2trk() -> None: ...
