from ..batteryrunners import BatteryRunner as BatteryRunner
from ..batteryrunners import Report as Report

def chk1(obj, fix: bool = False): ...
def chk2(obj, fix: bool = False): ...
def chk_warn(obj, fix: bool = False): ...
def chk_error(obj, fix: bool = False): ...
def test_init_basic() -> None: ...
def test_init_report() -> None: ...
def test_report_strings() -> None: ...
def test_logging() -> None: ...
def test_checks() -> None: ...
