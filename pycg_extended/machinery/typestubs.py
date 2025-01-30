# NOTE: Type based approximation
import ast
import importlib
import inspect
import os
import pickle
import re
import sys
import time
import types
from pathlib import Path

import jedi
import pygtrie
import requests
from pytype import module_utils
from pytype.pyi import parser
from pytype.pytd import pytd

from framework_models import check_alias

DATA_SCIECE_STUBS_DIR = os.path.join(
    Path(__file__).parent.absolute().parent.parent, "typestub-database"
)

ML_MODULES = {
    "tensorflow": os.path.join(DATA_SCIECE_STUBS_DIR, "tensorflow-stubs"),
    "pandas": os.path.join(DATA_SCIECE_STUBS_DIR, "pandas-stubs"),
    "numpy": os.path.join(DATA_SCIECE_STUBS_DIR, "numpy-stubs"),
    "matplotlib": os.path.join(DATA_SCIECE_STUBS_DIR, "matplotlib-stubs"),
    "keras": os.path.join(DATA_SCIECE_STUBS_DIR, "keras-stubs"),
    "sklearn": os.path.join(DATA_SCIECE_STUBS_DIR, "sklearn-stubs"),
    "statsmodels": os.path.join(DATA_SCIECE_STUBS_DIR, "statsmodels-stubs"),
    "seaborn": os.path.join(DATA_SCIECE_STUBS_DIR, "seaborn-stubs"),
    "nibabel": os.path.join(DATA_SCIECE_STUBS_DIR, "nibabel-stubs"),
}

type_stub_dirs = {
    k: os.path.join(DATA_SCIECE_STUBS_DIR, "typeshed", "stubs", k)
    for k in next(os.walk(os.path.join(DATA_SCIECE_STUBS_DIR, "typeshed", "stubs")))[1]
}

stdlib_stub_dirs = {
    k: os.path.join(DATA_SCIECE_STUBS_DIR, "typeshed", "stdlib", k)
    for k in next(os.walk(os.path.join(DATA_SCIECE_STUBS_DIR, "typeshed", "stdlib")))[1]
}

stdlib_stub_dirs = stdlib_stub_dirs | {
    "stdlib": os.path.join(DATA_SCIECE_STUBS_DIR, "stdlib")
}

ML_MODULES = ML_MODULES | type_stub_dirs | stdlib_stub_dirs
# "headergen", "pyright"
TYPE_INFERENCE_METHOD = "pyright"

SCRIPT_DIR = Path(__file__).parent.absolute()


def get_relative_module_name(filepath, module_name, module_path):
    module_string = []
    for i in range(1, len(filepath.parts) + 1):
        if filepath.parts[-i].startswith("__init__"):
            if str(filepath.parent) == module_path:
                module_string.append(module_name)
                break
        elif filepath.parts[-i] == Path(module_path).stem:
            module_string.append(module_name)
            break
        else:
            if filepath.parts[-i].endswith(".pyi"):
                module_string.append(filepath.stem)
            else:
                module_string.append(filepath.parts[-i])

    return ".".join(reversed(module_string))


def parse_pyi_file(filename, module_name, version=(3, 10)):
    with open(filename, "r") as f:
        src = f.read()

    out = parser.parse_pyi(src, filename, module_name, None)
    return out


PYTD_CACHE = {k: pygtrie.StringTrie(separator=".") for k in ML_MODULES}
if os.path.exists(os.path.join(SCRIPT_DIR, "pytd_cache.pickle")):
    PYTD_CACHE = pickle.load(open(os.path.join(SCRIPT_DIR, "pytd_cache.pickle"), "rb"))
else:
    for _module_name, _module_path in ML_MODULES.items():
        counter = 0
        for _file in Path(_module_path).rglob("*.pyi"):
            try:
                print(counter, _file)
                module_name = get_relative_module_name(
                    _file, _module_name, _module_path
                )
                PYTD_CACHE[_module_name][module_name] = parse_pyi_file(
                    _file, module_name
                )
            except Exception as e:
                print(f"Failed to read typestubs for: {_module_name}")
                print(e)

            counter += 1

    pickle.dump(
        PYTD_CACHE,
        open(os.path.join(SCRIPT_DIR, "pytd_cache.pickle"), "wb"),
    )


def get_qualname_from_text(input_dynamic_name):
    res = {"class": None, "func": None}
    try:
        input_str = inspect.unwrap(input_dynamic_name).__repr__()
        regex_class = r"(?<=of )(.*)(?= object)"

        if re.findall(regex_class, input_str):
            res["class"] = re.findall(regex_class, input_str)[0]

        regex_func = r"(?<=method )(.*)(?= of)"
        if re.findall(regex_func, input_str):
            res["func"] = re.findall(regex_func, input_str)[0]

        return res
    except Exception as e:
        return None


class TypeStubManager:
    RETURN_INFO_CACHE = {}

    def __init__(self, cache_pytb=True):
        self.inspect_module_imports = {}
        self.function_doc_strings = {}

        script_dir = Path(__file__).parent.absolute()

        self.functions_inspected = {}
        self.functions_info = {}

        self.pytd_cache = PYTD_CACHE

        self.already_tried_importing = []

    def load_library_into_memory(self, library):
        try:
            # if library in sys.modules:
            #     del sys.modules[library]
            self.inspect_module_imports[library] = importlib.import_module(library)
            # time.sleep(1)
            return True
        except ImportError:
            # print("Module not installed: ", library)
            return

    def get_inspect_info(self, mod_name, func_name):
        info = {}
        lib_name = func_name.split(".")[0]
        is_builtin = False

        if not (mod_name in self.inspect_module_imports or mod_name == "builtins"):
            return

        if mod_name == "builtins":
            is_builtin = True

        func_name = check_alias(func_name)
        mod_name = func_name.split(".")[0]
        regex = f"^({mod_name}?)"

        try:
            if is_builtin:
                _dynamic_name = eval(func_name)
            else:
                _dynamic_name = eval(
                    re.sub(
                        regex, f"self.inspect_module_imports['{mod_name}']", func_name
                    )
                )

            unwrapped = inspect.unwrap(_dynamic_name)

        except Exception as e:
            for _inspected in self.functions_inspected:
                if _inspected in func_name:
                    _func_name = func_name.replace(
                        _inspected, self.functions_inspected[_inspected]["api_call"]
                    )
                    _dynamic_name = eval(
                        re.sub(
                            regex,
                            f"self.inspect_module_imports['{mod_name}']",
                            _func_name,
                        )
                    )
                    unwrapped = inspect.unwrap(_dynamic_name)
                    break

        try:
            # if isinstance(inspect.unwrap(_dynamic_name), numpy.ufunc):
            #     print("ufunc: ", func_name)
            if isinstance(_dynamic_name, types.BuiltinFunctionType):
                info["module_name"] = mod_name
                info["qualified_name"] = unwrapped.__qualname__
                info["fullns"] = ".".join([info["module_name"], info["qualified_name"]])
                info["doc_string"] = inspect.getdoc(_dynamic_name)
                if info["module_name"] == "numpy":
                    if getattr(unwrapped, "__globals__", None):
                        info["module_name"] = getattr(unwrapped, "__globals__", None)[
                            "__name__"
                        ]
                    info["fullns"] = ".".join(
                        [info["module_name"], info["qualified_name"]]
                    )
            elif getattr(unwrapped, "__module__", None):
                info["module_name"] = unwrapped.__module__
                info["qualified_name"] = unwrapped.__qualname__
                info["fullns"] = ".".join([info["module_name"], info["qualified_name"]])
                info["doc_string"] = inspect.getdoc(_dynamic_name)
                if info["module_name"] == "numpy":
                    if getattr(unwrapped, "__globals__", None):
                        info["module_name"] = getattr(unwrapped, "__globals__", None)[
                            "__name__"
                        ]
                    info["fullns"] = ".".join(
                        [info["module_name"], info["qualified_name"]]
                    )

            elif getattr(unwrapped, "__objclass__", None):
                info["module_name"] = unwrapped.__objclass__.__module__
                info["qualified_name"] = unwrapped.__qualname__
                info["fullns"] = ".".join([info["module_name"], info["qualified_name"]])
                info["doc_string"] = inspect.getdoc(_dynamic_name)
            elif getattr(unwrapped, "__package__", None):
                info["module_name"] = unwrapped.__name__
                info["qualified_name"] = unwrapped.__name__
                info["fullns"] = unwrapped.__name__
                info["doc_string"] = inspect.getdoc(_dynamic_name)
            # elif isinstance(unwrapped, property):
            #     info["is_property"] = True
            else:
                info["module_name"] = unwrapped.__module__
                info["qualified_name"] = unwrapped.__qualname__
                info["doc_string"] = inspect.getdoc(_dynamic_name)

                full_name = get_qualname_from_text(_dynamic_name)
                if full_name:
                    if full_name["class"].startswith(lib_name):
                        info["fullns"] = ".".join(
                            [full_name["class"], full_name["func"]]
                        )
                    else:
                        info["fullns"] = func_name  # keep original if nothing works
                else:
                    info["fullns"] = ".".join(
                        [info["module_name"], info["qualified_name"]]
                    )

            if info["doc_string"]:
                self.function_doc_strings[info["fullns"]] = inspect.cleandoc(
                    info["doc_string"]
                )

            if info["fullns"] not in self.functions_inspected:
                self.functions_inspected[info["fullns"]] = {
                    "api_call": func_name,
                    "info": info,
                }
            if func_name not in self.functions_info:
                self.functions_info[func_name] = info

            return info
        except Exception as e:
            # print("Inspect error: ", str(e))
            return

    def get_methods_of_class(self, class_name):
        info = {}
        lib_name = class_name.split(".")[0]

        if not lib_name in self.inspect_module_imports:
            return

        if lib_name == "numpy":
            regex = f"^({lib_name}?)"

            try:
                _dynamic_name = eval(
                    re.sub(
                        regex, f"self.inspect_module_imports['{lib_name}']", class_name
                    )
                )
                method_list = [
                    func
                    for func in dir(_dynamic_name)
                    if callable(getattr(_dynamic_name, func))
                ]

                return method_list
            except Exception as e:
                return []

        regex = f"^({lib_name}?)"

        # HACK: Should be dynamic. something is not working well here with imports
        # Currently only returns Dataframe methods
        return [
            "T",
            "_AXIS_LEN",
            "_AXIS_NAMES",
            "_AXIS_NUMBERS",
            "_AXIS_ORDERS",
            "_AXIS_REVERSED",
            "_AXIS_TO_AXIS_NUMBER",
            "_HANDLED_TYPES",
            "__abs__",
            "__add__",
            "__and__",
            "__annotations__",
            "__array__",
            "__array_priority__",
            "__array_ufunc__",
            "__array_wrap__",
            "__bool__",
            "__class__",
            "__contains__",
            "__copy__",
            "__deepcopy__",
            "__delattr__",
            "__delitem__",
            "__dict__",
            "__dir__",
            "__divmod__",
            "__doc__",
            "__eq__",
            "__finalize__",
            "__floordiv__",
            "__format__",
            "__ge__",
            "__getattr__",
            "__getattribute__",
            "__getitem__",
            "__getstate__",
            "__gt__",
            "__hash__",
            "__iadd__",
            "__iand__",
            "__ifloordiv__",
            "__imod__",
            "__imul__",
            "__init__",
            "__init_subclass__",
            "__invert__",
            "__ior__",
            "__ipow__",
            "__isub__",
            "__iter__",
            "__itruediv__",
            "__ixor__",
            "__le__",
            "__len__",
            "__lt__",
            "__matmul__",
            "__mod__",
            "__module__",
            "__mul__",
            "__ne__",
            "__neg__",
            "__new__",
            "__nonzero__",
            "__or__",
            "__pos__",
            "__pow__",
            "__radd__",
            "__rand__",
            "__rdivmod__",
            "__reduce__",
            "__reduce_ex__",
            "__repr__",
            "__rfloordiv__",
            "__rmatmul__",
            "__rmod__",
            "__rmul__",
            "__ror__",
            "__round__",
            "__rpow__",
            "__rsub__",
            "__rtruediv__",
            "__rxor__",
            "__setattr__",
            "__setitem__",
            "__setstate__",
            "__sizeof__",
            "__str__",
            "__sub__",
            "__subclasshook__",
            "__truediv__",
            "__weakref__",
            "__xor__",
            "_accessors",
            "_accum_func",
            "_add_numeric_operations",
            "_agg_by_level",
            "_agg_examples_doc",
            "_agg_summary_and_see_also_doc",
            "_align_frame",
            "_align_series",
            "_arith_method",
            "_as_manager",
            "_box_col_values",
            "_can_fast_transpose",
            "_check_inplace_and_allows_duplicate_labels",
            "_check_inplace_setting",
            "_check_is_chained_assignment_possible",
            "_check_label_or_level_ambiguity",
            "_check_setitem_copy",
            "_clear_item_cache",
            "_clip_with_one_bound",
            "_clip_with_scalar",
            "_cmp_method",
            "_combine_frame",
            "_consolidate",
            "_consolidate_inplace",
            "_construct_axes_dict",
            "_construct_axes_from_arguments",
            "_construct_result",
            "_constructor",
            "_constructor_sliced",
            "_convert",
            "_count_level",
            "_data",
            "_dir_additions",
            "_dir_deletions",
            "_dispatch_frame_op",
            "_drop_axis",
            "_drop_labels_or_levels",
            "_ensure_valid_index",
            "_find_valid_index",
            "_from_arrays",
            "_from_mgr",
            "_get_agg_axis",
            "_get_axis",
            "_get_axis_name",
            "_get_axis_number",
            "_get_axis_resolvers",
            "_get_block_manager_axis",
            "_get_bool_data",
            "_get_cleaned_column_resolvers",
            "_get_column_array",
            "_get_index_resolvers",
            "_get_item_cache",
            "_get_label_or_level_values",
            "_get_numeric_data",
            "_get_value",
            "_getitem_bool_array",
            "_getitem_multilevel",
            "_gotitem",
            "_hidden_attrs",
            "_indexed_same",
            "_info_axis",
            "_info_axis_name",
            "_info_axis_number",
            "_info_repr",
            "_init_mgr",
            "_inplace_method",
            "_internal_names",
            "_internal_names_set",
            "_is_copy",
            "_is_homogeneous_type",
            "_is_label_or_level_reference",
            "_is_label_reference",
            "_is_level_reference",
            "_is_mixed_type",
            "_is_view",
            "_iset_item",
            "_iset_item_mgr",
            "_iset_not_inplace",
            "_iter_column_arrays",
            "_ixs",
            "_join_compat",
            "_logical_func",
            "_logical_method",
            "_maybe_cache_changed",
            "_maybe_update_cacher",
            "_metadata",
            "_min_count_stat_function",
            "_needs_reindex_multi",
            "_protect_consolidate",
            "_reduce",
            "_reindex_axes",
            "_reindex_columns",
            "_reindex_index",
            "_reindex_multi",
            "_reindex_with_indexers",
            "_replace_columnwise",
            "_repr_data_resource_",
            "_repr_fits_horizontal_",
            "_repr_fits_vertical_",
            "_repr_html_",
            "_repr_latex_",
            "_reset_cache",
            "_reset_cacher",
            "_sanitize_column",
            "_series",
            "_set_axis",
            "_set_axis_name",
            "_set_axis_nocheck",
            "_set_is_copy",
            "_set_item",
            "_set_item_frame_value",
            "_set_item_mgr",
            "_set_value",
            "_setitem_array",
            "_setitem_frame",
            "_setitem_slice",
            "_slice",
            "_stat_axis",
            "_stat_axis_name",
            "_stat_axis_number",
            "_stat_function",
            "_stat_function_ddof",
            "_take_with_is_copy",
            "_to_dict_of_blocks",
            "_typ",
            "_update_inplace",
            "_validate_dtype",
            "_values",
            "_where",
            "abs",
            "add",
            "add_prefix",
            "add_suffix",
            "agg",
            "aggregate",
            "align",
            "all",
            "any",
            "append",
            "apply",
            "applymap",
            "asfreq",
            "asof",
            "assign",
            "astype",
            "as_matrix",
            "at",
            "at_time",
            "attrs",
            "axes",
            "backfill",
            "between_time",
            "bfill",
            "bool",
            "boxplot",
            "clip",
            "columns",
            "combine",
            "combine_first",
            "compare",
            "convert_dtypes",
            "copy",
            "corr",
            "corrwith",
            "count",
            "cov",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "describe",
            "diff",
            "div",
            "divide",
            "dot",
            "drop",
            "drop_duplicates",
            "droplevel",
            "dropna",
            "dtypes",
            "duplicated",
            "empty",
            "eq",
            "equals",
            "eval",
            "ewm",
            "expanding",
            "explode",
            "ffill",
            "fillna",
            "filter",
            "first",
            "first_valid_index",
            "flags",
            "floordiv",
            "from_csv",
            "from_dict",
            "from_records",
            "ge",
            "get",
            "groupby",
            "gt",
            "head",
            "hist",
            "iat",
            "idxmax",
            "idxmin",
            "iloc",
            "index",
            "infer_objects",
            "info",
            "insert",
            "interpolate",
            "isin",
            "isna",
            "isnull",
            "items",
            "iteritems",
            "iterrows",
            "itertuples",
            "join",
            "keys",
            "kurt",
            "kurtosis",
            "last",
            "last_valid_index",
            "le",
            "loc",
            "lookup",
            "lt",
            "mad",
            "mask",
            "max",
            "mean",
            "median",
            "melt",
            "memory_usage",
            "merge",
            "min",
            "mod",
            "mode",
            "mul",
            "multiply",
            "ndim",
            "ne",
            "nlargest",
            "notna",
            "notnull",
            "nsmallest",
            "nunique",
            "pad",
            "pct_change",
            "pipe",
            "pivot",
            "pivot_table",
            "plot",
            "pop",
            "pow",
            "prod",
            "product",
            "quantile",
            "query",
            "radd",
            "rank",
            "rdiv",
            "reindex",
            "reindex_like",
            "rename",
            "rename_axis",
            "reorder_levels",
            "replace",
            "resample",
            "reset_index",
            "rfloordiv",
            "rmod",
            "rmul",
            "rolling",
            "round",
            "rpow",
            "rsub",
            "rtruediv",
            "sample",
            "select_dtypes",
            "sem",
            "set_axis",
            "set_flags",
            "set_index",
            "shape",
            "shift",
            "size",
            "skew",
            "slice_shift",
            "sort_index",
            "sort_values",
            "sparse",
            "squeeze",
            "stack",
            "std",
            "style",
            "sub",
            "subtract",
            "sum",
            "swapaxes",
            "swaplevel",
            "tail",
            "take",
            "to_clipboard",
            "to_csv",
            "to_dict",
            "to_excel",
            "to_feather",
            "to_gbq",
            "to_hdf",
            "to_html",
            "to_json",
            "to_latex",
            "to_markdown",
            "to_numpy",
            "to_parquet",
            "to_period",
            "to_pickle",
            "to_records",
            "to_sql",
            "to_stata",
            "to_string",
            "to_timestamp",
            "to_xarray",
            "to_xml",
            "transform",
            "transpose",
            "truediv",
            "truncate",
            "tshift",
            "tz_convert",
            "tz_localize",
            "unstack",
            "update",
            "value_counts",
            "values",
            "var",
            "where",
            "xs",
        ]

    def parse_pyi_file(self, filename, module_name, version=(3, 6)):
        with open(filename, "r") as f:
            src = f.read()

        # out, _ = parser.parse_pyi_debug(
        #     src, filename, module_name, version, None)

        out = parser.parse_pyi(src, filename, module_name, None)
        return out

    def get_relative_module_name(self, filepath, module_name, module_path):
        module_string = []
        for i in range(1, len(filepath.parts) + 1):
            if filepath.parts[-i].startswith("__init__"):
                if str(filepath.parent) == module_path:
                    module_string.append(module_name)
                    break
            elif filepath.parts[-i] == Path(module_path).stem:
                module_string.append(module_name)
                break
            else:
                if filepath.parts[-i].endswith(".pyi"):
                    module_string.append(filepath.stem)
                else:
                    module_string.append(filepath.parts[-i])

        return ".".join(reversed(module_string))

    def recursive_pytd_lookup(self, module_name, inspect_info):
        try:
            _pyi_match = self.pytd_cache[module_name].longest_prefix(
                inspect_info["module_name"]
            )
            _full_ns = inspect_info["fullns"]
        except Exception as e:
            # Check stdlib
            _pyi_match = self.pytd_cache["stdlib"].longest_prefix(
                f"stdlib.{inspect_info['module_name']}"
            )
            _full_ns = f"stdlib.{inspect_info['fullns']}"

        if "." in inspect_info["qualified_name"]:
            # _pyi_match = self.pytd_cache[module_name].longest_prefix(
            #     inspect_info["module_name"]
            # )
            combos = []
            difference = [
                item
                for item in _full_ns.split(".")
                if item not in _pyi_match.key.split(".")
            ][:-1]
            if difference:
                split_func_name = _full_ns.split(".")[-1]
                _suffix = []
                for i in range(1, len(difference) + 1):
                    _suffix.append(".".join(difference[:i]))

                combos.extend(["{}.{}".format(_pyi_match.key, x) for x in _suffix])
            # else:
            #     combos.append(inspect_info['fullns'])
            #     _res = _pyi_match.value.Lookup(split_func_name)
            #     return _res

            for _combo in combos:
                try:
                    _res = _pyi_match.value.Lookup(_combo).Lookup(split_func_name)
                    return _res
                except:
                    continue

            # print()
        else:
            _res = _pyi_match.value.Lookup(_full_ns)
            return _res

    def is_builtin(self, name):
        return name in __builtins__

    def lookup_return_type_hg(self, func_name, module_imports, **kwargs):
        def get_pandas_namedtypes(named_type):
            _pyi = self.pytd_cache["pandas"].longest_prefix("pandas._typing")
            _type = _pyi.value.Lookup(named_type)
            if isinstance(_type, pytd.TypeParameter):
                return [_type.bound]
            elif isinstance(_type, pytd.UnionType):
                return _pyi.value.Lookup(named_type).type.type_list
            else:
                return []

        def get_numpy_namedtypes(named_type):
            _pyi = self.pytd_cache["numpy"].longest_prefix("numpy.typing")
            _type = _pyi.value.Lookup(named_type).type
            return _type

        func_name = check_alias(func_name)

        hashed_ref = f"{func_name}"
        # if hashed_ref in TypeStubManager.RETURN_INFO_CACHE:
        #     return TypeStubManager.RETURN_INFO_CACHE[hashed_ref]

        res_dict = {}
        module_name = func_name.split(".")[0]

        parent = func_name.split(".")[0]
        # if not parent in self.pytd_cache:
        #     return None

        self.inspect_module_imports = module_imports
        call_attrs = func_name.split(".")[1:-1]
        combos = [parent, func_name]

        for i in range(len(call_attrs)):
            _joined = ".".join(call_attrs[: i + 1])
            # find_api_string = ".".join([parent,_joined])
            # if find_api_string in self.functions_inspected:
            #     combos.append(self.functions_inspected[find_api_string]['api_call'])
            combos.append(".".join([parent, _joined]))

        _res = None
        inspect_info = None

        if self.is_builtin(func_name):
            inspect_info = self.get_inspect_info("builtins", func_name)
            try:
                _res = (
                    self.pytd_cache["stdlib"]
                    .longest_prefix(f"stdlib.builtins.{func_name}")
                    .value.Lookup(f"stdlib.builtins.{func_name}")
                )
            except:
                pass

        else:
            for _combo in combos:
                try:
                    if _combo not in self.inspect_module_imports:
                        # HACK: find how to handle importing main lib
                        # self.load_library_into_memory(_combo)
                        try:
                            if _combo not in self.inspect_module_imports:
                                if _combo not in self.already_tried_importing:
                                    self.inspect_module_imports[
                                        _combo
                                    ] = importlib.import_module(_combo)
                                    importlib.invalidate_caches()
                        except Exception as e:
                            if _combo not in self.already_tried_importing:
                                self.already_tried_importing.append(_combo)
                                print(e)
                            continue

                    inspect_info = self.get_inspect_info(_combo, func_name)
                    # if "count" in inspect_info["fullns"]:
                    #     print()
                    try:
                        _pyi_match = self.pytd_cache[module_name].longest_prefix(
                            inspect_info["module_name"]
                        )
                    except Exception as e:
                        # Check stdlib
                        _pyi_match = self.pytd_cache["stdlib"].longest_prefix(
                            f"stdlib.{inspect_info['module_name']}"
                        )

                    _res = self.recursive_pytd_lookup(module_name, inspect_info)
                    if not _res:
                        continue
                    # _res = _pyi_match.value.Lookup(inspect_info['fullns'])
                    break
                except Exception as e:
                    continue

        if _res:
            if isinstance(_res, pytd.Function):
                # HACK: only the first match?
                _return_types = []
                for _def in _res.signatures:
                    # _def = _res.signatures[0]
                    if isinstance(_def.return_type, pytd.NamedType):
                        if _def.return_type.name.startswith("pandas._typing"):
                            for _ret_type in get_pandas_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type.name == "NoneType":
                                    _return_types.append(_ret_type.name)
                        else:
                            if not _def.return_type.name == "NoneType":
                                if _def.return_type.name.startswith("stdlib.builtins."):
                                    _return_types.append(
                                        _def.return_type.name.split("stdlib.builtins.")[
                                            1
                                        ]
                                    )
                                else:
                                    _return_types.append(_def.return_type.name)
                    elif isinstance(_def.return_type, pytd.UnionType):
                        for _ret_type in _def.return_type.type_list:
                            _return_types.append(_ret_type.name)
                    elif isinstance(_def.return_type, pytd.TypeParameter):
                        if _def.return_type.bound:
                            _return_types.append(_def.return_type.bound.name)
                    elif isinstance(_def.return_type, pytd.AnythingType):
                        # _return_types.append(_def.return_type.bound.name)
                        pass
                    elif isinstance(_def.return_type, pytd.GenericType):
                        if _def.return_type.name.startswith("numpy.typing"):
                            for _ret_type in get_numpy_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type == "NoneType":
                                    _return_types.append(_ret_type)
                        else:
                            # TODO: Return multiple types
                            # if _def.return_type.name == "tuple":
                            #     for d in _def.return_type.parameters:
                            #         _return_types.append(d.name)
                            # else:
                            #     _return_types.append(_def.return_type.name)
                            _return_types.append(_def.return_type.name)
                res_dict = {
                    "return_type": list(set(_return_types)),
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Alias):
                # TODO: should also handle UnionType for return_type here
                res_dict = {
                    "return_type": [_res.type.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Class):
                res_dict = {
                    "return_type": [_res.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Constant):
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            else:
                res_dict = None

        else:
            for _combo in combos:
                try:
                    inspect_info = self.get_inspect_info(_combo, func_name)
                except Exception as e:
                    pass
                    # ignore
                if inspect_info:
                    break
            if inspect_info:
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": None,
                    "doc_string": inspect_info["doc_string"],
                }

        if hashed_ref not in TypeStubManager.RETURN_INFO_CACHE:
            TypeStubManager.RETURN_INFO_CACHE[hashed_ref] = res_dict

        return res_dict

    def lookup_return_type_jedi(
        self, func_name, module_imports, node=None, filename=None
    ):
        self.inspect_module_imports = module_imports

        def get_pandas_namedtypes(named_type):
            _pyi = self.pytd_cache["pandas"].longest_prefix("pandas._typing")
            _type = _pyi.value.Lookup(named_type)
            if isinstance(_type, pytd.TypeParameter):
                return [_type.bound]
            elif isinstance(_type, pytd.UnionType):
                return _pyi.value.Lookup(named_type).type.type_list
            else:
                return []

        def get_numpy_namedtypes(named_type):
            _pyi = self.pytd_cache["numpy"].longest_prefix("numpy.typing")
            _type = _pyi.value.Lookup(named_type).type
            return _type

        hashed_ref = f"{func_name}_{hash(node)}_{hash(filename)}"
        if hashed_ref in TypeStubManager.RETURN_INFO_CACHE:
            return TypeStubManager.RETURN_INFO_CACHE[hashed_ref]

        res_dict_jedi = {}
        res_dict = {}

        if filename and node:
            # if isinstance(node, ast.Import):
            #     print("")
            # elif isinstance(node, ast.ImportFrom):
            #     print("")
            # elif isinstance(node, ast.Attribute):
            #     print("")

            # print(node)
            try:
                code = open(filename).read()

                # Get jedi names
                # hashed_ref_names = f"jedi_names_{hash(filename)}"
                # if hashed_ref_names in TypeStubManager.RETURN_INFO_CACHE:
                #     jedi_ref_names = TypeStubManager.RETURN_INFO_CACHE[hashed_ref_names]
                # else:
                #     jedi_names = jedi.Script(code).get_names(all_scopes=1, references=1)
                #     refs_var_names = {}
                #     for _r in jedi_names:
                #         refs_var_names[f"{_r.line}_{_r.column}:{_r.name}"] = {
                #             "line": _r.line,
                #             "column": _r.column,
                #         }

                #     jedi_ref_names = {}

                #     for var, pos in refs_var_names.items():
                #         _infer = jedi.Script(code).infer(
                #             pos["line"], pos["column"], prefer_stubs=1
                #         )
                #         for inferred in _infer:
                #             if var not in jedi_ref_names:
                #                 jedi_ref_names[var] = []

                #             if inferred.full_name not in jedi_ref_names[var]:
                #                 jedi_ref_names[var].append(inferred.full_name)

                #     TypeStubManager.RETURN_INFO_CACHE[hashed_ref_names] = jedi_ref_names

                _infer = jedi.Script(code).infer(
                    node.lineno, node.end_col_offset, prefer_stubs=1
                )
                ret_types = []
                ret_name = None
                ret_type_of_def = None
                ret_doc_string = None
                for _name in _infer:
                    if not ret_name:
                        ret_name = _name.full_name
                        ret_type_of_def = _name.type
                        ret_doc_string = _name.docstring(raw=1)
                    # TODO: can cache?
                    # _signature = _name.get_type_hint()
                    _types = _name.execute()
                    for _t in _types:
                        if _t.full_name not in ret_types:
                            if _t.full_name.startswith("builtins."):
                                if _t.full_name != "builtins.NoneType":
                                    ret_types.append(_t.full_name.split("builtins.")[1])
                            else:
                                ret_types.append(_t.full_name)

                res_dict_jedi = {
                    "return_type": list(set(ret_types)),
                    "return_name": ret_name,
                    "type_of_def": ret_type_of_def,
                    "doc_string": ret_doc_string,
                }
                # if ret_types and ret_name:
                #     return res_dict
                # else:
                #     print("Need to inspect!", func_name)
            except Exception as e:
                print("Jedi error")
                print(e)
                res_dict_jedi = {}

        else:
            print()
        if hashed_ref not in TypeStubManager.RETURN_INFO_CACHE:
            TypeStubManager.RETURN_INFO_CACHE[hashed_ref] = res_dict_jedi

        # uncomment to deactivate debugging mode
        # if all([not bool(v) for v in res_dict_jedi.values()]):
        #     return None
        # else:
        #     return res_dict_jedi

        module_name = func_name.split(".")[0]

        parent = func_name.split(".")[0]
        # if not parent in self.pytd_cache:
        #     return None

        self.inspect_module_imports = module_imports
        call_attrs = func_name.split(".")[1:-1]
        combos = [parent, func_name]

        for i in range(len(call_attrs)):
            _joined = ".".join(call_attrs[: i + 1])
            # find_api_string = ".".join([parent,_joined])
            # if find_api_string in self.functions_inspected:
            #     combos.append(self.functions_inspected[find_api_string]['api_call'])
            combos.append(".".join([parent, _joined]))

        _res = None
        inspect_info = None
        for _combo in combos:
            try:
                if _combo not in self.inspect_module_imports:
                    # HACK: find how to handle importing main lib
                    # self.load_library_into_memory(_combo)
                    continue

                inspect_info = self.get_inspect_info(_combo, func_name)
                # if "count" in inspect_info["fullns"]:
                #     print()
                _pyi_match = self.pytd_cache[module_name].longest_prefix(
                    inspect_info["module_name"]
                )
                _res = self.recursive_pytd_lookup(module_name, inspect_info)
                if not _res:
                    continue
                # _res = _pyi_match.value.Lookup(inspect_info['fullns'])
                break
            except Exception as e:
                continue

        if _res:
            if isinstance(_res, pytd.Function):
                # HACK: only the first match?
                _return_types = []
                for _def in _res.signatures:
                    # _def = _res.signatures[0]
                    if isinstance(_def.return_type, pytd.NamedType):
                        if _def.return_type.name.startswith("pandas._typing"):
                            for _ret_type in get_pandas_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type.name == "NoneType":
                                    _return_types.append(_ret_type.name)
                        else:
                            if not _def.return_type.name == "NoneType":
                                _return_types.append(_def.return_type.name)
                    elif isinstance(_def.return_type, pytd.UnionType):
                        for _ret_type in _def.return_type.type_list:
                            _return_types.append(_ret_type.name)
                    elif isinstance(_def.return_type, pytd.TypeParameter):
                        _return_types.append(_def.return_type.bound.name)
                    elif isinstance(_def.return_type, pytd.AnythingType):
                        # _return_types.append(_def.return_type.bound.name)
                        pass
                    elif isinstance(_def.return_type, pytd.GenericType):
                        if _def.return_type.name.startswith("numpy.typing"):
                            for _ret_type in get_numpy_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type == "NoneType":
                                    _return_types.append(_ret_type)
                        else:
                            _return_types.append(_def.return_type.name)
                res_dict = {
                    "return_type": list(set(_return_types)),
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Alias):
                # TODO: should also handle UnionType for return_type here
                res_dict = {
                    "return_type": [_res.type.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Class):
                res_dict = {
                    "return_type": [_res.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Constant):
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            else:
                res_dict = None

        else:
            if inspect_info:
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": None,
                    "doc_string": inspect_info["doc_string"],
                }

        if res_dict_jedi == res_dict:
            print("#>>> Result Equal!", func_name)
            print("\n>>>###############\n\n")
        elif res_dict_jedi.get("return_type") == res_dict.get("return_type"):
            if res_dict_jedi.get("return_name") == res_dict.get("return_name"):
                if res_dict_jedi.get("type_of_def") == res_dict.get("type_of_def"):
                    print("#>>> Almost Equal!, but not doc", func_name)
                    print("\n>>>###############\n\n")
                    res_dict_jedi["doc_string"] = res_dict["doc_string"]

        else:
            if not res_dict and all([not bool(v) for v in res_dict_jedi.values()]):
                print("##### Result BOTH null!", func_name)
            else:
                print("##### Result Diff!", func_name)
                print("\nres_dict_jedi")
                print("return_type", res_dict_jedi.get("return_type"))
                print("return_name", res_dict_jedi.get("return_name"))
                print("type_of_def", res_dict_jedi.get("type_of_def"))

                print("\nres_dict")
                print("return_type", res_dict.get("return_type"))
                print("return_name", res_dict.get("return_name"))
                print("type_of_def", res_dict.get("type_of_def"))

                print("\n###############\n\n")

        if hashed_ref not in TypeStubManager.RETURN_INFO_CACHE:
            TypeStubManager.RETURN_INFO_CACHE[hashed_ref] = res_dict

        if all([not bool(v) for v in res_dict_jedi.values()]):
            return None
        else:
            return res_dict_jedi

    def lookup_return_type_pyright(
        self, func_name, module_imports, node=None, filename=None
    ):
        self.inspect_module_imports = module_imports

        def get_pandas_namedtypes(named_type):
            _pyi = self.pytd_cache["pandas"].longest_prefix("pandas._typing")
            _type = _pyi.value.Lookup(named_type)
            if isinstance(_type, pytd.TypeParameter):
                return [_type.bound]
            elif isinstance(_type, pytd.UnionType):
                return _pyi.value.Lookup(named_type).type.type_list
            else:
                return []

        def get_numpy_namedtypes(named_type):
            _pyi = self.pytd_cache["numpy"].longest_prefix("numpy.typing")
            _type = _pyi.value.Lookup(named_type).type
            return _type

        hashed_ref = f"{func_name}_{hash(node)}_{hash(filename)}"
        if hashed_ref in TypeStubManager.RETURN_INFO_CACHE:
            return TypeStubManager.RETURN_INFO_CACHE[hashed_ref]

        res_dict_jedi = {}
        res_dict = {}

        # uncomment to deactivate debugging mode
        # if all([not bool(v) for v in res_dict_jedi.values()]):
        #     return None
        # else:
        #     return res_dict_jedi

        module_name = func_name.split(".")[0]

        parent = func_name.split(".")[0]
        # if not parent in self.pytd_cache:
        #     return None

        self.inspect_module_imports = module_imports
        call_attrs = func_name.split(".")[1:-1]
        combos = [parent, func_name]

        for i in range(len(call_attrs)):
            _joined = ".".join(call_attrs[: i + 1])
            # find_api_string = ".".join([parent,_joined])
            # if find_api_string in self.functions_inspected:
            #     combos.append(self.functions_inspected[find_api_string]['api_call'])
            combos.append(".".join([parent, _joined]))

        _res = None
        inspect_info = None
        for _combo in combos:
            try:
                if _combo not in self.inspect_module_imports:
                    # HACK: find how to handle importing main lib
                    # self.load_library_into_memory(_combo)
                    continue

                inspect_info = self.get_inspect_info(_combo, func_name)
                # if "count" in inspect_info["fullns"]:
                #     print()
                _pyi_match = self.pytd_cache[module_name].longest_prefix(
                    inspect_info["module_name"]
                )
                _res = self.recursive_pytd_lookup(module_name, inspect_info)
                if not _res:
                    continue
                # _res = _pyi_match.value.Lookup(inspect_info['fullns'])
                break
            except Exception as e:
                continue

        if _res:
            if isinstance(_res, pytd.Function):
                # HACK: only the first match?
                _return_types = []
                for _def in _res.signatures:
                    # _def = _res.signatures[0]
                    if isinstance(_def.return_type, pytd.NamedType):
                        if _def.return_type.name.startswith("pandas._typing"):
                            for _ret_type in get_pandas_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type.name == "NoneType":
                                    _return_types.append(_ret_type.name)
                        else:
                            if not _def.return_type.name == "NoneType":
                                _return_types.append(_def.return_type.name)
                    elif isinstance(_def.return_type, pytd.UnionType):
                        for _ret_type in _def.return_type.type_list:
                            _return_types.append(_ret_type.name)
                    elif isinstance(_def.return_type, pytd.TypeParameter):
                        _return_types.append(_def.return_type.bound.name)
                    elif isinstance(_def.return_type, pytd.AnythingType):
                        # _return_types.append(_def.return_type.bound.name)
                        pass
                    elif isinstance(_def.return_type, pytd.GenericType):
                        if _def.return_type.name.startswith("numpy.typing"):
                            for _ret_type in get_numpy_namedtypes(
                                _def.return_type.name
                            ):
                                if not _ret_type == "NoneType":
                                    _return_types.append(_ret_type)
                        else:
                            _return_types.append(_def.return_type.name)
                res_dict = {
                    "return_type": list(set(_return_types)),
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Alias):
                # TODO: should also handle UnionType for return_type here
                res_dict = {
                    "return_type": [_res.type.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Class):
                res_dict = {
                    "return_type": [_res.name],
                    "return_name": _res.name,
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            elif isinstance(_res, pytd.Constant):
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": type(_res).__name__.lower(),
                    "doc_string": inspect_info["doc_string"],
                }
            else:
                res_dict = None

        else:
            if inspect_info:
                res_dict = {
                    "return_type": None,
                    "return_name": inspect_info["fullns"],
                    "type_of_def": None,
                    "doc_string": inspect_info["doc_string"],
                }

        if filename and node:
            print(node)
            try:
                _url = "http://0.0.0.0:8088/?file_path={file_path}&lineno={lineno}&col_offset={col_offset}&func_name={func_name}"
                x = requests.get(
                    _url.format(
                        file_path=filename,
                        lineno=node.lineno - 1,
                        col_offset=node.end_col_offset - 1,
                        func_name=func_name,
                    )
                )

                if x.json()["data"]:
                    print(x.json()["data"])
                    _type = x.json()["data"]
                else:
                    _type = []

                res_dict_jedi = {
                    "return_type": _type,
                    "return_name": res_dict["return_name"],
                    "type_of_def": res_dict["type_of_def"],
                    "doc_string": res_dict["doc_string"],
                }

                if isinstance(_res, (pytd.Class, pytd.Alias)):
                    res_dict_jedi["return_type"] = res_dict["return_type"]

                # if ret_types and ret_name:
                #     return res_dict
                # else:
                #     print("Need to inspect!", func_name)
            except Exception as e:
                print("Jedi error")
                print(e)
                res_dict_jedi = {}

        if res_dict_jedi == res_dict:
            print("#>>> Result Equal!", func_name)
            print("\n>>>###############\n\n")
        elif res_dict_jedi.get("return_type") == res_dict.get("return_type"):
            if res_dict_jedi.get("return_name") == res_dict.get("return_name"):
                if res_dict_jedi.get("type_of_def") == res_dict.get("type_of_def"):
                    print("#>>> Almost Equal!, but not doc", func_name)
                    print("\n>>>###############\n\n")

        else:
            if not res_dict and all([not bool(v) for v in res_dict_jedi.values()]):
                print("##### Result BOTH null!", func_name)
            else:
                print("##### Result Diff!", func_name)
                print("\nres_dict_jedi")
                print("return_type", res_dict_jedi.get("return_type"))
                print("return_name", res_dict_jedi.get("return_name"))
                print("type_of_def", res_dict_jedi.get("type_of_def"))

                print("\nres_dict")
                print("return_type", res_dict.get("return_type"))
                print("return_name", res_dict.get("return_name"))
                print("type_of_def", res_dict.get("type_of_def"))

                print("\n###############\n\n")

        if hashed_ref not in TypeStubManager.RETURN_INFO_CACHE:
            TypeStubManager.RETURN_INFO_CACHE[hashed_ref] = res_dict_jedi

        if all([not bool(v) for v in res_dict_jedi.values()]):
            return None
        else:
            return res_dict_jedi

    def lookup_return_type_dummy(
        self, func_name, module_imports, node=None, filename=None
    ):
        return None

    # lookup_return_type = lookup_return_type_dummy
    # lookup_return_type = lookup_return_type_jedi
    lookup_return_type = lookup_return_type_hg
    # lookup_return_type = lookup_return_type_pyright
