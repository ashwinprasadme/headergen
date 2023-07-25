import argparse
import ast
import builtins
import glob
import json
import logging
import os
import shutil
from collections import OrderedDict, deque
from pathlib import Path
from pprint import pprint

import simplejson as sjson

import headergen.utils as utils
from framework_models import MODELS as PIPELINE_LIBRARY_MODEL
from framework_models import PHASES as PIPELINE_PHASES
from framework_models import lookup_pipeline_tag, lookup_pipeline_tag_ml
from pycg_extended import formats as pycg_formats
from pycg_extended import pycg

# CallGraphGenerator

# TODO: this could actually be done while travering AST
BUILTIN_FUNC_LIST = dir(builtins)
MAX_ITER = 500

ML_PIPELINE_MODEL = True


def get_dl_pipeline_tag(function_call, doc_string=None):
    if ML_PIPELINE_MODEL:
        return lookup_pipeline_tag_ml(function_call, doc_string)
    else:
        return lookup_pipeline_tag(function_call)


def sort_pycg_calls(analysis_info, main_file_name):
    imports_info = analysis_info["imports_info"]
    function_list = analysis_info["func_calls"]
    context_func_calls = analysis_info["context_func_calls"]
    defined_func_info = analysis_info["defined_func_info"]

    # TODO: Add case where library is not aliased
    library_calls = {}
    unknown_calls = {}
    builtin_calls = {}
    defined_calls = {}

    def get_tag_info(func):
        if func.startswith("<builtin>."):
            _tag = {
                "func_call": func,
                "dl_pipeline_tag": [PIPELINE_PHASES["BUILTIN_FUNCTION"]],
            }

        # Sort defined func calls
        elif func.startswith(main_file_name):
            _tag = {
                "func_call": func,
                "dl_pipeline_tag": [PIPELINE_PHASES["DECLARED_FUNCTION"]],
            }

        # Sort library calls
        else:
            if ML_PIPELINE_MODEL:
                if func in analysis_info["function_doc_strings"]:
                    _tags = get_dl_pipeline_tag(
                        func, analysis_info["function_doc_strings"][func]
                    )
                else:
                    print(f"No doc string for: {func}")
                    _tags = []

                _tag = {"func_call": func, "dl_pipeline_tag": _tags}
            else:
                _tag = {"func_call": func, "dl_pipeline_tag": get_dl_pipeline_tag(func)}

        if func in analysis_info["function_doc_strings"]:
            _tag["doc_string"] = analysis_info["function_doc_strings"][func]

        return _tag

    # Sort main body calls
    for _context_key, _context_value in context_func_calls.items():
        for _func_i in range(len(_context_value)):
            _tag = get_tag_info(_context_value[_func_i]["func_call"])
            context_func_calls[_context_key][_func_i].update(_tag)

    # Sort funtiondef body calls
    for _def_func, _def_func_body in defined_func_info.items():
        if _def_func not in defined_calls:
            defined_calls[_def_func] = []
        for _func in _def_func_body:
            if (main_file_name in _func) and ("." in _func):
                # ignore own definition
                continue
            _tag = get_tag_info(_func)

            defined_calls[_def_func].append(_tag)

    return {
        "context_library_calls": dict(sorted(context_func_calls.items())),
        "defined_calls": defined_calls,
        "function_doc_strings": analysis_info["function_doc_strings"],
        "pycg_output": analysis_info["pycg_output"],
        "pycg_cg": analysis_info["pycg_cg"],
        "imports_info": imports_info,
        "eag": analysis_info["eag"],
        "line_uses": analysis_info["line_uses"],
        "call_args": analysis_info["call_args"],
        "locals_types": analysis_info["locals_types"],
        "types_formatted": analysis_info["types_formatted"],
    }


def get_pycg_analysis(py_ntbk_path):
    main_file_name = Path(py_ntbk_path).name.split(".")[0]

    defined_func_info = {}
    imports_info = []

    cg = pycg.CallGraphGenerator(
        [py_ntbk_path], str(Path(py_ntbk_path).parent), MAX_ITER, "call-site"
    )
    cg.analyze()

    formatter = pycg_formats.Simple(cg)
    ag_formatter = pycg_formats.AsGraph(cg)
    cs_formatter = pycg_formats.CallSites(cg)

    cg_json = cs_formatter.get_cg()
    imports_json = cs_formatter.get_imports()
    imports_info = list(imports_json[main_file_name]["imports"])

    ag_json = ag_formatter.generate()
    cs_json = cs_formatter.generate(module_name=main_file_name)

    call_args = cg.ca

    pycg_output = cg.output_call_sites(module_name=Path(py_ntbk_path).stem)

    context_func_calls = {}
    for _line, _calls in cs_json.items():
        context_func_calls[_line] = []
        for _c in _calls:
            context_func_calls[_line].append({"func_call": _c})

    for _key in cg_json:
        # get main functions --> func_calls (should be context_func_calls)
        if main_file_name == _key:
            # print("func_calls")
            pass
            # pprint(cg_json[main_file_name])

        # get functions inside definitions, classes, nested --> defined_func_info
        if (main_file_name in _key) and ("." in _key):
            # pprint(cg_json[_key])
            defined_func_info[_key] = cg_json[_key]

    types_formatted = []

    # Class vars
    for _class_var_fact in cg.def_manager.class_vars:
        _class_var = _class_var_fact["name"]
        local_name = None
        for _scp, variables in cg.state["scopes"].items():
            for _v in variables:
                if _v.endswith(_class_var):
                    local_name = _v
                    break
            if local_name:
                break

        local_defs = cg.def_manager.transitive_closure().get(local_name)
        if local_defs:
            _type_fact = {"file": Path(py_ntbk_path).name, "type": []}
            for _def in local_defs:
                if _def in cg.def_manager.defs:
                    # Prepare type_fact dict
                    _type_fact["line_number"] = _class_var_fact["lineno"]

                    if _scp != main_file_name:
                        _type_fact["function"] = _scp.split(":")[0].split(
                            f"{main_file_name}."
                        )[1]

                    _type_fact["variable"] = utils.get_clear_all_lineno(_class_var)

                    if cg.def_manager.defs[_def].def_type == "EXTERNALDEF":
                        pass
                        # _type_fact["type"].append(utils.get_clear_lineno(cg.def_manager.defs[_def].fullns))
                        # types_formatted.append(_type_fact)

                    elif cg.def_manager.defs[_def].def_type == "FUNCTIONDEF":
                        _type_fact["type"].append("callable")
                        types_formatted.append(_type_fact)

                    elif cg.def_manager.defs[_def].get_lit_pointer().values:
                        for lit_value in (
                            cg.def_manager.defs[_def].get_lit_pointer().values
                        ):
                            _type_fact["type"].append(type(lit_value).__name__)

                            types_formatted.append(_type_fact)

    # return types
    for _def_name, _def_value in cg.def_manager.defs.items():
        if _def_name.endswith(".<RETURN>"):
            _type_fact = {
                "file": Path(py_ntbk_path).name,
                "line_number": utils.get_last_lineno_return(_def_name),
                "function": utils.get_clear_all_lineno(_def_name)
                .split(f"{main_file_name}.")[-1]
                .split(".<RETURN>")[0],
                "type": [],
            }

            local_defs = cg.def_manager.transitive_closure().get(_def_name)
            if local_defs:
                for _def in local_defs:
                    if _def in cg.def_manager.defs:
                        if cg.def_manager.defs[_def].def_type == "EXTERNALDEF":
                            pass
                            # _type_fact["type"].append(utils.get_clear_lineno(cg.def_manager.defs[_def].fullns))
                            # types_formatted.append(_type_fact)

                        elif cg.def_manager.defs[_def].def_type == "FUNCTIONDEF":
                            _type_fact["type"].append("callable")
                            types_formatted.append(_type_fact)

                        elif cg.def_manager.defs[_def].get_lit_pointer().values:
                            for lit_value in (
                                cg.def_manager.defs[_def].get_lit_pointer().values
                            ):
                                _type_fact["type"].append(type(lit_value).__name__)

                                types_formatted.append(_type_fact)

    def decode_type():
        pass

    # local types
    locals_types = {}
    for _local_fact in cg.def_manager.locals_defs:
        _local = _local_fact["name"]
        _type_fact = {"file": Path(py_ntbk_path).name, "type": []}
        try:
            # find scope
            local_name = None
            for _scp, variables in cg.state["scopes"].items():
                for _v in variables:
                    if _v.endswith(_local):
                        local_name = _v
                        break
                if local_name:
                    break

            if not local_name:
                locals_types[local_name] = []
                continue

            if local_name not in locals_types:
                locals_types[local_name] = []
            local_defs = cg.def_manager.transitive_closure().get(local_name)
            if local_defs:
                for _def in local_defs:
                    if _def in cg.def_manager.defs:
                        # Prepare type_fact dict
                        _type_fact["line_number"] = _local_fact["lineno"]

                        if _scp != main_file_name:
                            _type_fact["function"] = _scp.split(":")[0].split(
                                f"{main_file_name}."
                            )[1]

                        if _local_fact["node_type"] == "param":
                            _type_fact["parameter"] = _local_fact["id"]
                        else:
                            _type_fact["variable"] = _local_fact["id"]

                        if cg.def_manager.defs[_def].def_type == "EXTERNALDEF":
                            locals_types[local_name].append(
                                utils.get_clear_lineno(cg.def_manager.defs[_def].fullns)
                            )
                            # _type_fact["type"].append(utils.get_clear_lineno(cg.def_manager.defs[_def].fullns))
                            # types_formatted.append(_type_fact)

                        elif cg.def_manager.defs[_def].def_type == "FUNCTIONDEF":
                            _type_fact["type"].append("callable")
                            types_formatted.append(_type_fact)

                        elif cg.def_manager.defs[_def].get_lit_pointer().values:
                            for lit_value in (
                                cg.def_manager.defs[_def].get_lit_pointer().values
                            ):
                                _type_fact["type"].append(type(lit_value).__name__)

                                types_formatted.append(_type_fact)

                                locals_types[local_name].append(
                                    type(lit_value).__name__
                                )

                        elif cg.def_manager.defs[_def].def_type == "NAMEDEF":
                            if dict_num := utils.is_dict(
                                cg.def_manager.defs[_def].fullns
                            ):
                                _type_fact["type"].append("dict")
                                types_formatted.append(_type_fact)
                                locals_types[local_name].append("dict")
                                all_dicts = [
                                    x
                                    for x in cg.def_manager.defs
                                    if utils.is_dict(x, specific_dict=dict_num)
                                ]
                                # TODO: Add all dict keys to type list
                            elif utils.is_list(cg.def_manager.defs[_def].fullns):
                                _type_fact["type"].append("list")
                                types_formatted.append(_type_fact)
                                locals_types[local_name].append("list")

        except Exception as e:
            print(f"Failed return_type fetch! {str(e)}")
            continue

    return {
        "func_calls": sorted(cg_json[main_file_name]),
        "context_func_calls": context_func_calls,
        "pycg_cg": cg_json,
        "pycg_output": pycg_output,
        "eag": cg.state["defs"],
        "line_uses": cg.def_manager.line_uses,
        "defined_func_info": defined_func_info,
        "imports_info": imports_info,
        "function_doc_strings": cg.typestub_manager.function_doc_strings,
        "locals_types": locals_types,
        "types_formatted": types_formatted,
        "call_args": call_args,
    }


def get_annotated_analysis(tree, py_ntbk_path=None):
    main_file_name = Path(py_ntbk_path).name.split(".")[0]

    pycg_analysis_info = get_pycg_analysis(py_ntbk_path)
    # pprint(pycg_analysis_info)

    library_calls_pycg = sort_pycg_calls(pycg_analysis_info, main_file_name)

    return library_calls_pycg


# Direct run
if __name__ == "__main__":
    pass
