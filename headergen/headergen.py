# Read line-by-line, check type of node, get function information based on node
# %%
import collections
import glob
import logging
import os
import shutil
import time
from collections import deque
from pathlib import Path
from pprint import pprint

import gast as ast
import jupytext
import simplejson as json
from intervaltree import Interval, IntervalTree

import headergen.static_analysis_helpers as sa_helpers
import headergen.utils as utils
from framework_models import PHASE_GROUPS
from framework_models import PHASES as PIPELINE_PHASES
from framework_models import get_high_level_phase, lookup_pipeline_tag
from headergen.node_visitor import HeaderGenVisitor
from framework_models.ml_function_classifier.CellClassifier import CellClassifier


# %%
def find_first_block_start(py_source):
    py_source_split = py_source.split("\n")
    lineno = 1
    for _line in py_source_split:
        if _line.startswith("# %%"):
            break

        lineno += 1

    return lineno


# Find all blocks and the line numbers in notebook script
# TODO: Should be a better way to do this
def find_block_numbers(py_source):
    py_source_split = py_source.split("\n")
    _start, _end = None, None
    lineno = 1
    block = 1
    mapping = {}

    _current_md = False
    for _line in py_source_split:
        if _line.startswith("# %%"):
            if _start is None:
                _start = lineno
                if _line.startswith("# %% [markdown]"):
                    _current_md = True
                else:
                    _current_md = False
            else:
                _end = lineno
                if _end == (_start + 1):
                    _start = lineno
                    continue

                if not _current_md:
                    mapping[block] = {"start": _start, "end": _end - 1}
                    block += 1

                _start = _end
                if _line.startswith("# %% [markdown]"):
                    _current_md = True
                else:
                    _current_md = False

        lineno += 1

    if not _current_md:
        mapping[block] = {"start": _start, "end": lineno - 1}
    return mapping


def find_code_block_numbers(py_source):
    py_source_split = py_source.split("\n")
    _start, _end = None, None
    lineno = 1
    block = 1
    code_block = 1
    mapping = {}

    _current_md = False
    for _line in py_source_split:
        if _line.startswith("# %%"):
            if _start is None:
                _start = lineno
                if _line.startswith("# %% [markdown]"):
                    _current_md = True
                else:
                    _current_md = False
            else:
                _end = lineno
                if _end == (_start + 1):
                    _start = lineno
                    continue

                if not _current_md:
                    mapping[code_block] = block
                    code_block += 1

                block += 1

                _start = _end
                if _line.startswith("# %% [markdown]"):
                    _current_md = True
                else:
                    _current_md = False

        lineno += 1

    if not _current_md:
        mapping[code_block] = block
    return mapping


def get_block_of_lineno(lineno, block_mapping):
    for map_key, map_value in block_mapping.items():
        if map_value["start"] <= lineno <= map_value["end"]:
            return map_key

    return None


def get_library_classified(doc_string):
    classified = {}
    for _ds_func, _ds_string in doc_string.items():
        _lib = _ds_func.split(".")[0]
        if _lib not in classified:
            classified[_lib] = {}

        classified[_lib][_ds_func] = _ds_string

    return classified


def add_markdown_cell_to_source(
    py_ntbk, block_mapping, tag, lineno, block_no, doc_string
):
    # TODO: Can handle first insert with 3 to avoid empty cell?
    start_block = get_block_of_lineno(lineno, block_mapping)

    top_doc_string = "<details><summary style='list-style: none; cursor: pointer;'><u>View function calls</u></summary>\n<ul>\n\n{}\n\n</ul>\n</details>"

    lib_list = "<li> <strong class='hglib'>{}</strong>\n<ul>\n{}\n</ul>\n</li>"
    ind_list = "<li>\n<details><summary style='list-style: none; cursor: pointer;'><u>{}</u></summary>\n<blockquote>\n<code>\n{}\n\n</code>\n<a href='#{}'>back to header</a>\n</blockquote>\n</details>\n</li>"

    # lib_list =
    classified_funcs = get_library_classified(doc_string)

    _lib_list = []
    for _lib, _calls in classified_funcs.items():
        _calls_template = []
        for _c, _ds in _calls.items():
            _calls_template.append(ind_list.format(_c, _ds, block_no))

        _lib_list.append(lib_list.format(_lib, "\n".join(_calls_template)))

    if _lib_list:
        markdown_block = [
            "",
            '# %% [markdown] deletable=false editable=false run_control={"frozen": true}',
            "# <div> <h3 class='hg'>{blkno}. {tag}</h3>  <a id='{blkno}'></a><small><a href='#top_phases'>back to top</a></small>{ds} </div>".format(
                tag=tag, blkno=block_no, ds=top_doc_string.format("\n".join(_lib_list))
            ),
            "",
        ]
    else:
        markdown_block = [
            "",
            '# %% [markdown] deletable=false editable=false run_control={"frozen": true}',
            "# <div> <h3 class='hg'>{blkno}. {tag}</h3>  <a id='{blkno}'></a><small><a href='#top_phases'>back to top</a></small> </div>".format(
                tag=tag, blkno=block_no
            ),
            "",
        ]
    py_ntbk = py_ntbk.split("\n")
    py_ntbk = py_ntbk[: lineno - 1] + markdown_block + py_ntbk[lineno - 1 :]

    py_ntbk = "\n".join(py_ntbk)

    new_mapping = find_block_numbers(py_ntbk)
    for _block_key, _block_value in block_mapping.items():
        if "dl_pipeline_tag" not in block_mapping[_block_key]:
            continue
        new_mapping[_block_key]["dl_pipeline_tag"] = block_mapping[_block_key][
            "dl_pipeline_tag"
        ]
        new_mapping[_block_key]["doc_string"] = block_mapping[_block_key]["doc_string"]
        new_mapping[_block_key]["call_args"] = block_mapping[_block_key]["call_args"]
        new_mapping[_block_key]["dl_pipeline_tag_counter"] = block_mapping[_block_key][
            "dl_pipeline_tag_counter"
        ]
        new_mapping[_block_key]["function_order"] = block_mapping[_block_key][
            "function_order"
        ]

    return py_ntbk, new_mapping, _lib_list


def deep_sorted(obj, *, key=None, reverse=False):
    if isinstance(obj, dict):
        return {
            k: deep_sorted(v, key=key, reverse=reverse)
            for k, v in sorted(obj.items(), key=key, reverse=reverse)
        }
    if isinstance(obj, list):
        return [
            deep_sorted(v, key=key, reverse=reverse)
            for i, v in sorted(enumerate(obj), key=key, reverse=reverse)
        ]
    return obj


def add_phase_info_to_source(
    py_ntbk, block_mapping, ml_phases_data, imports_info, function_call_doc_strings
):
    lineno = find_first_block_start(py_ntbk)
    start_block = 1
    markdown_block = [
        "",
        '# %% [markdown] deletable=false editable=false source_hidden=true run_control={"frozen": true}',
        "# ### Index of ML Operations<a id='top_phases'></a>",
        "<div><ul>",
    ]
    top_phase_string = "<ul><li><details><summary style='list-style: none; cursor: pointer;'><strong>{phase}</strong></summary>\n<ul>\n\n{cell_list}\n\n</ul>\n</details></li></ul>"
    top_phase_string_empty = "<ul><li><details><summary style='list-style: none;'><s>{phase}</s> (no calls found)</summary>\n<ul>\n\n{cell_list}\n\n</ul>\n</details></li></ul>"
    top_phase_string_high_level = "<li><details><summary style='list-style: none; cursor: pointer;'><h3><span style='color:#42a5f5'>{phase}</span></h3></summary>\n<ul>\n\n{cell_list}\n\n</ul>\n</details></li>"
    top_phase_string_high_level_empty = "<li><details><summary style='list-style: none;'><h3><span style='color:#42a5f5'>{phase}</span></h3></summary>\n<ul>\n\n{cell_list}\n\n</ul>\n</details></li>"
    cell_lib_string = "<li><details open><summary style='list-style: none; cursor: pointer;'><strong><u>Cell # {cell_id}</u></strong></summary><small><a href=#{cell_id}>goto cell # {cell_id}</a></small>\n<ul>\n\n{lib_list}\n\n</ul>\n</details></li>"
    cell_lib_string_empty = "<li><details><summary style='list-style: none;'><b>Cell # {cell_id}</b></summary><small><a href=#{cell_id}>goto cell # {cell_id}</a></small>\n<i>No function calls found in the cell</i>\n</details></li>"
    phase_all_lib_string = "<li><details><summary style='list-style: none; cursor: pointer;'><u>{cell_id}</u></summary>\n<ul>\n\n{lib_list}\n\n</ul>\n</details></li>"
    # cell_list = "<li> <a href='#{cell_id}'><b>Cell # {cell_id}</b></a></li>"

    top_doc_string = "<details><summary style='list-style: none; cursor: pointer;'><strong>View All ML API Calls in Notebook</strong></summary>\n<ul>\n\n{}\n\n</ul>\n</details>"

    lib_list = "<li> <b>{}</b>\n<ul>\n{}\n</ul>\n</li>"
    ind_list = "<li>\n<details><summary style='list-style: none; cursor: pointer;'><u>{}</u></summary>\n<blockquote>\n<code>\n{}\n\n</code>\n<a href='#top_phases'>back to header</a>\n</blockquote>\n</details>\n</li>"
    ind_args_list_empty = "<li>\n<details><summary style='list-style: none; cursor: pointer;'><u>{}</u> | {} </summary>\n<blockquote>\n<code>\n{}\n\n</code>\n<a href='#top_phases'>back to header</a>\n</blockquote>\n</details>\n</li>"
    ind_args_list = "<li>\n<details><summary style='list-style: none; cursor: pointer;'><u>{}</u> | <b>(See Args)</b> </summary> {}\n<blockquote>\n<code>\n{}\n\n</code>\n<a href='#top_phases'>back to header</a>\n</blockquote>\n</details>\n</li>"

    selected_phases = [
        # PIPELINE_PHASES["LIBRARY_LOADING"],
        PIPELINE_PHASES["VISUALIZATION"],
        PIPELINE_PHASES["DATA_CLEANING_PREPARATION"],
        PIPELINE_PHASES["DATA_PROFILING_AND_EXPLORATORY_DATA_ANALYSIS"],
        PIPELINE_PHASES["DATA_CLEANING_FILTERING"],
        PIPELINE_PHASES["DATA_SUB_SAMPLING_AND_TRAIN_TEST_SPLITTING"],
        PIPELINE_PHASES["FEATURE_ENGINEERING"],
        PIPELINE_PHASES["FEATURE_TRANSFORMATION"],
        PIPELINE_PHASES["FEATURE_SELECTION"],
        PIPELINE_PHASES["MODEL_BUILDING_AND_TRAINING"],
        PIPELINE_PHASES["MODEL_TRAINING"],
        PIPELINE_PHASES["MODEL_PARAMETER_TUNING"],
        PIPELINE_PHASES["MODEL_VALIDATION_AND_ASSEMBLING"],
    ]

    GROUP_PHASES = False

    phase_cell_mapping = {k: [] for k in selected_phases}
    for _block_key, _block_value in block_mapping.items():
        if "dl_pipeline_tag" not in block_mapping[_block_key]:
            continue
        if GROUP_PHASES:
            phases = [
                get_high_level_phase(_p)
                for _p in block_mapping[_block_key]["dl_pipeline_tag"]
            ]
        else:
            phases = block_mapping[_block_key]["dl_pipeline_tag"]

        for _phase in selected_phases:
            if _phase in phases:
                # if _phase in _block_value["dl_pipeline_tag"]:
                phase_cell_mapping[_phase].append(_block_key)
                # Also add to high-level category
                high_level_phase = get_high_level_phase(_phase)
                if high_level_phase in phase_cell_mapping:
                    if _block_key not in phase_cell_mapping[high_level_phase]:
                        phase_cell_mapping[high_level_phase].append(_block_key)
                # print(_phase, "in", _block_key)

    # Add imports info to markdown
    imports_info = list(set([x.split(".")[0] for x in imports_info]))
    markdown_block.append(
        top_phase_string.format(
            phase="Imported Libraries",
            cell_list="\n".join(
                ["<li><b>{}</b></li>".format(x) for x in sorted(imports_info)]
            ),
        )
    )

    def get_args_func(func, cell):
        nonlocal block_mapping
        args_list = []
        if cell in block_mapping:
            for k_lineno, v_funcs in block_mapping[cell]["call_args"].items():
                if func in v_funcs:
                    if any([v_funcs[func]["args"], v_funcs[func]["kwargs"]]):
                        args_list.append(v_funcs[func])

        return args_list

    def get_filtered_lib_list(function_list, phase, cell):
        # lib_list =
        classified_funcs = get_library_classified(function_list)

        _lib_list = []
        for _lib, _calls in classified_funcs.items():
            _calls_template = []
            for _c, _ds in _calls.items():
                if phase in lookup_pipeline_tag(_c):
                    args_list = get_args_func(_c, cell)
                    if args_list:
                        _str = "\n".join(
                            [
                                f"<ul><li><b>Args:</b> {x['args']} | <b>Kwargs:</b> {x['kwargs']}</li></ul>"
                                for x in args_list
                            ]
                        )
                        _calls_template.append(ind_args_list.format(_c, _str, _ds))
                        # _str = "\n".join([f"<ul><li><b>Args:</b> {x['args']} </li><li> <b>Kwargs:</b> {x['kwargs']}</li></ul>" for x in args_list])
                    else:
                        _str = "(No Args Found)"
                        _calls_template.append(
                            ind_args_list_empty.format(_c, _str, _ds)
                        )

            if _calls_template:
                _lib_list.append(lib_list.format(_lib, "\n".join(_calls_template)))

        return _lib_list

    for _phase, _phase_cells in phase_cell_mapping.items():
        _cell_list = []
        _all_phase_calls = {}
        for _cell in _phase_cells:
            filtered_calls = None
            if _cell in ml_phases_data:
                _all_phase_calls = _all_phase_calls | block_mapping[_cell]["doc_string"]
                if ml_phases_data[_cell]:
                    filtered_calls = get_filtered_lib_list(
                        block_mapping[_cell]["doc_string"], _phase, _cell
                    )
                    if filtered_calls:
                        _cell_list.append(
                            cell_lib_string.format(
                                cell_id=_cell, lib_list="\n".join(filtered_calls)
                            )
                        )
                    else:
                        _cell_list.append(
                            cell_lib_string.format(
                                cell_id=_cell, lib_list="Code pattern match"
                            )
                        )

                        # It is a pattern match
            #     else:
            #         _cell_list.append(cell_lib_string_empty.format(cell_id=_cell))
            # else:
            #     _cell_list.append(cell_lib_string_empty.format(cell_id=_cell))

        if _all_phase_calls:
            all_calls_filtered = get_filtered_lib_list(_all_phase_calls, _phase, _cell)
            if all_calls_filtered:
                _cell_list.insert(
                    0,
                    phase_all_lib_string.format(
                        cell_id=f'View All "{_phase}" Calls',
                        lib_list="\n".join(all_calls_filtered),
                    ),
                )

        if _cell_list:
            if _phase in [
                PIPELINE_PHASES["DATA_CLEANING_PREPARATION"],
                PIPELINE_PHASES["FEATURE_ENGINEERING"],
                PIPELINE_PHASES["MODEL_BUILDING_AND_TRAINING"],
            ]:
                markdown_block.append(
                    top_phase_string_high_level.format(
                        phase=_phase, cell_list="\n".join(_cell_list)
                    )
                )
            else:
                markdown_block.append(
                    top_phase_string.format(
                        phase=_phase, cell_list="\n".join(_cell_list)
                    )
                )
        else:
            if _phase in [
                PIPELINE_PHASES["DATA_CLEANING_PREPARATION"],
                PIPELINE_PHASES["FEATURE_ENGINEERING"],
                PIPELINE_PHASES["MODEL_BUILDING_AND_TRAINING"],
            ]:
                markdown_block.append(
                    top_phase_string_high_level_empty.format(
                        phase=_phase, cell_list="None"
                    )
                )
            else:
                markdown_block.append(
                    top_phase_string_empty.format(phase=_phase, cell_list="None")
                )

    # close high level list
    markdown_block.append("</ul>\n<hr>\n")

    sorted_function_calls = deep_sorted(
        get_library_classified(function_call_doc_strings)
    )

    _lib_list = []
    for _lib, _calls in sorted_function_calls.items():
        _calls_template = []
        for _c, _ds in _calls.items():
            _calls_template.append(ind_list.format(_c, _ds))

        _lib_list.append(lib_list.format(_lib, "\n".join(_calls_template)))

    if _lib_list:
        markdown_block.append(top_doc_string.format("\n".join(_lib_list)))

    markdown_block.append("</div>")
    markdown_block.append("")

    py_ntbk = py_ntbk.split("\n")
    py_ntbk = py_ntbk[: lineno - 1] + markdown_block + py_ntbk[lineno - 1 :]

    for _block_key, _block_value in block_mapping.items():
        if int(_block_key) >= int(start_block):
            block_mapping[_block_key]["start"] = _block_value["start"] + len(
                markdown_block
            )
            block_mapping[_block_key]["end"] = _block_value["end"] + len(markdown_block)

    py_ntbk = "\n".join(py_ntbk)

    return py_ntbk, block_mapping


def add_toc_to_source(py_ntbk, block_mapping, toc_data):
    lineno = find_first_block_start(py_ntbk)
    start_block = 1
    markdown_block = [
        "",
        '# %% [markdown] deletable=false editable=false run_control={"frozen": true}',
        "# # Table of Contents <a id='top_toc'></a>",
    ]

    prev_toc = ""
    for _b_k, _b_v in toc_data.items():
        if prev_toc == _b_v:
            markdown_block.append(
                "#    + <sub><sup>[{}. {}](#{})</sub></sup>".format(_b_k, _b_v, _b_k)
            )
        else:
            markdown_block.append("# + [{}. {}](#{})".format(_b_k, _b_v, _b_k))
        prev_toc = _b_v
        # print("+ [{}](#{})".format(_b_k, _b_v))

    markdown_block.append("")

    py_ntbk = py_ntbk.split("\n")
    py_ntbk = py_ntbk[: lineno - 1] + markdown_block + py_ntbk[lineno - 1 :]

    for _block_key, _block_value in block_mapping.items():
        if int(_block_key) >= int(start_block):
            block_mapping[_block_key]["start"] = _block_value["start"] + len(
                markdown_block
            )
            block_mapping[_block_key]["end"] = _block_value["end"] + len(markdown_block)

    py_ntbk = "\n".join(py_ntbk)

    return py_ntbk, block_mapping


def get_cell_summaries(py_ntbk, hg_visitor):
    # Summarize blocks
    block_mapping = find_block_numbers(py_ntbk)

    cell_callsites_mapping = {}
    # HACK: can be done better?
    for _block_key, _block_value in block_mapping.items():
        block_mapping[_block_key]["dl_pipeline_tag"] = []
        block_mapping[_block_key]["doc_string"] = {}
        block_mapping[_block_key]["call_args"] = {}
        block_mapping[_block_key]["function_order"] = []

        cell_callsites_mapping[_block_key] = set()
        _lineno = _block_value["start"]
        while _lineno in range(_block_value["start"], _block_value["end"] + 1):
            if _lineno in hg_visitor.context_library_calls:
                for _f_order in hg_visitor.context_library_calls[_lineno]:
                    block_mapping[_block_key]["function_order"].append(
                        _f_order["func_call"]
                    )

            if _lineno not in hg_visitor.source_code_tags:
                # print("Lineno not in headergen visitor: ", str(_lineno))
                _lineno = _lineno + 1
                continue

            # if not line number is inside a body interval
            if not bool(hg_visitor.body_intervals.at(_lineno)):
                if _lineno in hg_visitor.source_code_tags:
                    # HACK: to match and retreive latest function summary that includes imports information
                    if (
                        PIPELINE_PHASES["DECLARED_FUNCTION"]
                        in hg_visitor.source_code_tags[_lineno]["dl_pipeline_tag"]
                    ):
                        for _c_func in hg_visitor.context_library_calls[_lineno]:
                            if (
                                not PIPELINE_PHASES["DECLARED_FUNCTION"]
                                in _c_func["dl_pipeline_tag"]
                            ):
                                cell_callsites_mapping[_block_key].add(
                                    _c_func["func_call"]
                                )
                                block_mapping[_block_key]["dl_pipeline_tag"].extend(
                                    _c_func["dl_pipeline_tag"]
                                )
                                if (
                                    _c_func["func_call"]
                                    in hg_visitor.source_code_tags[_lineno][
                                        "doc_string"
                                    ]
                                ):
                                    block_mapping[_block_key]["doc_string"][
                                        _c_func["func_call"]
                                    ] = hg_visitor.source_code_tags[_lineno][
                                        "doc_string"
                                    ][
                                        _c_func["func_call"]
                                    ]

                            else:
                                _part_name = _c_func["func_call"].split(".")[-1]
                                if _part_name in hg_visitor.defined_function_summaries:
                                    block_mapping[_block_key]["call_args"] = (
                                        block_mapping[_block_key]["call_args"]
                                        | hg_visitor.defined_function_summaries[
                                            _part_name
                                        ]["call_args"]
                                    )

                    else:
                        if _lineno in hg_visitor.context_library_calls:
                            for _func in hg_visitor.context_library_calls[_lineno]:
                                cell_callsites_mapping[_block_key].add(
                                    _func["func_call"]
                                )
                        block_mapping[_block_key]["dl_pipeline_tag"].extend(
                            hg_visitor.source_code_tags[_lineno]["dl_pipeline_tag"]
                        )
                        for _func_name, _func_ds in hg_visitor.source_code_tags[
                            _lineno
                        ]["doc_string"].items():
                            block_mapping[_block_key]["doc_string"][
                                _func_name
                            ] = _func_ds

                        if _lineno in hg_visitor.call_args_line_no:
                            block_mapping[_block_key]["call_args"][_lineno] = (
                                hg_visitor.call_args_line_no[_lineno]
                            )

            else:
                if (
                    PIPELINE_PHASES["FUNCTION_DEFINITION"]
                    in hg_visitor.source_code_tags[_lineno]["dl_pipeline_tag"]
                ):
                    for _interval in hg_visitor.body_intervals.at(_lineno):
                        _part_name = _interval.data["name"]
                        if _part_name in hg_visitor.defined_function_summaries:
                            block_mapping[_block_key]["call_args"] = (
                                block_mapping[_block_key]["call_args"]
                                | hg_visitor.defined_function_summaries[_part_name][
                                    "call_args"
                                ]
                            )

                            if _interval.data["node_type"] == "FunctionDef":
                                block_mapping[_block_key]["dl_pipeline_tag"].extend(
                                    [PIPELINE_PHASES["FUNCTION_DEFINITION"]]
                                )
                                block_mapping[_block_key]["dl_pipeline_tag"].extend(
                                    hg_visitor.defined_function_summaries[
                                        _interval.data["name"]
                                    ]["dl_pipeline_tag"]
                                )
                                block_mapping[_block_key]["doc_string"] = (
                                    block_mapping[_block_key]["doc_string"]
                                    | hg_visitor.defined_function_summaries[
                                        _interval.data["name"]
                                    ]["doc_string"]
                                )
                                # Skip function definition lines
                                _lineno = _interval[1]

                # not func def, someother body
                else:
                    if _lineno in hg_visitor.context_library_calls:
                        for _func in hg_visitor.context_library_calls[_lineno]:
                            cell_callsites_mapping[_block_key].add(_func["func_call"])

                        for _func_name, _func_ds in hg_visitor.source_code_tags[
                            _lineno
                        ]["doc_string"].items():
                            block_mapping[_block_key]["doc_string"][
                                _func_name
                            ] = _func_ds

                    block_mapping[_block_key]["dl_pipeline_tag"].extend(
                        hg_visitor.source_code_tags[_lineno]["dl_pipeline_tag"]
                    )

            _lineno = _lineno + 1

        # No tag found, try in pattern based tags
        tag_list = list(set(block_mapping[_block_key]["dl_pipeline_tag"]))
        if "Builtin Function" in tag_list:
            tag_list.remove("Builtin Function")
        if "Unknown" in tag_list:
            tag_list.remove("Unknown")
        if not tag_list:
            for _l in range(_block_value["start"], _block_value["end"] + 1):
                if _l in hg_visitor.pattern_matches:
                    block_mapping[_block_key]["dl_pipeline_tag"].extend(
                        hg_visitor.pattern_matches[_l]["dl_pipeline_tag"]
                    )

        block_mapping[_block_key]["dl_pipeline_tag_counter"] = collections.Counter(
            block_mapping[_block_key]["dl_pipeline_tag"]
        )

    return block_mapping, cell_callsites_mapping


def get_analysis_output(nb_path, out_path="."):
    # Read ipynb and convert to python for analysis
    file_name = Path(nb_path).stem
    if Path(nb_path).suffix == ".py":
        py_ntbk_path = nb_path
        py_ntbk = open(py_ntbk_path).read()
    elif Path(nb_path).suffix == ".ipynb":
        ntbk = jupytext.read(nb_path)
        py_ntbk = jupytext.writes(ntbk, fmt="py:percent")
        py_ntbk_path = "{}/{}-hg-analysis.py".format(Path(nb_path).parent, file_name)
        # write to python file for analysis
        # TODO: what about other files in the same directory?
        jupytext.write(ntbk, py_ntbk_path, fmt="py:percent")

    else:
        return "File not supported!"

    # Read Source, gen AST and get analysis
    # TODO: Replace this analysis with PyCG
    tree = ast.parse(py_ntbk)
    analysis_info = sa_helpers.get_analysis_output(tree, py_ntbk_path)

    return analysis_info


def start_headergen(nb_path, out_path=".", debug_mode=False, create_linted_file=True, start_cellclassifer=False):
    # Read ipynb and convert to python for analysis
    
    
    # implement the cellclassifier logic here
    # Do classification or something here
    
    if start_cellclassifer:
        cell_classifier = CellClassifier()
        # Ashwin I need your help here:
        # The usage of the cell_classifier is as follows:
        # cell_classifier.predict_workflow_step(cell_code)
        # It will first check if there a folder named cell_classifier exists, 
        # if not it will create one and then download the pkl file to it 
        # and then load it in the classifer attribute of the object
        # The predict_workflow_step method will take a cell code as input and return the predicted workflow step - The names can be found in /framework_models/ml_function_classifier/CellClassifier.py
        # I have added the code for the classifier in headergen, but I require your help to integrate it with the headergen code, where the predictions can be used to classify the cells
        
    file_name = Path(nb_path).stem
    if Path(nb_path).suffix == ".py":
        py_ntbk_path = nb_path
        py_ntbk = open(py_ntbk_path).read()
    elif Path(nb_path).suffix == ".ipynb":
        
        ntbk = jupytext.read(nb_path)
        py_ntbk = jupytext.writes(ntbk, fmt="py:percent")
        py_ntbk_path = "{}/{}-hg-analysis.py".format(Path(nb_path).parent, file_name)
        # write to python file for analysis
        # TODO: what about other files in the same directory?
        jupytext.write(ntbk, py_ntbk_path, fmt="py:percent")

    else:
        return "File not supported!"

    # Read Source, gen AST and get analysis
    # TODO: Replace this analysis with PyCG
    tree = ast.parse(py_ntbk)
    analysis_info = sa_helpers.get_annotated_analysis(tree, py_ntbk_path)

    if Path(nb_path).suffix == ".py":
        analysis_info["file_name"] = file_name
    elif Path(nb_path).suffix == ".ipynb":
        analysis_info["file_name"] = f"{file_name}-hg-analysis"

    hg_visitor = HeaderGenVisitor(analysis_info)
    hg_visitor.visit(tree)

    # %%
    if Path(nb_path).suffix == ".ipynb":
        block_mapping, cell_callsites_mapping = get_cell_summaries(py_ntbk, hg_visitor)
        code_block_mapping = find_code_block_numbers(py_ntbk)

        if create_linted_file:
            py_ntbk_linted, block_mapping_linted = py_ntbk, block_mapping

            toc_data = {}
            ml_phases_data = {}

            _header_template = "<a id='the_destination'></a>"
            filtered_tags = [
                PIPELINE_PHASES["BUILTIN_FUNCTION"],
                PIPELINE_PHASES["DECLARED_FUNCTION"],
                PIPELINE_PHASES["UNKNOWN"],
                PIPELINE_PHASES["FUNCTION_DEFINITION"],
            ]
            for _block_key, _block_value in block_mapping_linted.items():
                # print(_block_key)
                # _top_tag = block_mapping_linted[_block_key]["dl_pipeline_tag_counter"].most_common(1)[0][0]
                # Build heading string from dict with filtering
                high_level_phases = [
                    get_high_level_phase(_phase)
                    for _phase in block_mapping_linted[_block_key]["dl_pipeline_tag"]
                ]
                _top_tag = " | ".join(
                    f"{key}"
                    for key in sorted(set(high_level_phases))
                    if key not in filtered_tags
                )
                # Markdown tag with function call count
                # _top_tag = " | ".join(f'{get_high_level_phase(key)} ({value})' for key, value in dict(block_mapping_linted[_block_key]["dl_pipeline_tag_counter"].most_common()).items() if key not in filtered_tags)
                _lineno = block_mapping_linted[_block_key]["start"]
                if len(_top_tag) > 0:
                    # insert tag here
                    # print(_top_tag)
                    toc_data[_block_key] = _top_tag
                    (
                        py_ntbk_linted,
                        block_mapping_linted,
                        lib_list,
                    ) = add_markdown_cell_to_source(
                        py_ntbk_linted,
                        block_mapping_linted,
                        _top_tag,
                        _lineno,
                        _block_key,
                        _block_value["doc_string"],
                    )
                    ml_phases_data[_block_key] = lib_list

            # py_ntbk_linted, block_mapping_linted = add_toc_to_source(py_ntbk_linted, block_mapping_linted, toc_data)
            py_ntbk_linted, block_mapping_linted = add_phase_info_to_source(
                py_ntbk_linted,
                block_mapping_linted,
                ml_phases_data,
                analysis_info["imports_info"],
                analysis_info["function_doc_strings"],
            )

            # %%
            py_ntbk_linted = jupytext.reads(py_ntbk_linted, fmt="py:percent")
            # HACK: pop to clear empty cell created on top. Investigate why.
            py_ntbk_linted.cells.pop(0)

            jupytext.write(
                py_ntbk_linted,
                "{}/{}-linted.ipynb".format(out_path, file_name),
                fmt="ipynb",
            )

            if debug_mode:
                jupytext.write(
                    py_ntbk_linted,
                    "{}/{}-linted.py".format(out_path, file_name),
                    fmt="py:percent",
                )

        if not debug_mode:
            # remove interm file
            Path(py_ntbk_path).unlink(missing_ok=True)

        # Map block numbers to actual code blocks
        new_code_block_mapping = {}
        for block_no in block_mapping:
            new_code_block_mapping[code_block_mapping[block_no]] = block_mapping[
                block_no
            ]

        return_info = {
            "out_file": "{}/{}-linted.ipynb".format(out_path, file_name),
            "analysis_info": analysis_info,
            "source_code_tags": hg_visitor.source_code_tags,
            "cell_callsites": utils.get_cell_numbers(
                analysis_info["pycg_output"], str(nb_path), file_name
            ),
            "block_mapping": block_mapping_linted,
            "code_block_mapping": new_code_block_mapping,
        }

    elif Path(nb_path).suffix == ".py":
        for _line_no in hg_visitor.source_code_tags:
            tag_list = list(
                set(hg_visitor.source_code_tags[_line_no]["dl_pipeline_tag"])
            )
            if "Builtin Function" in tag_list:
                tag_list.remove("Builtin Function")
            if "Unknown" in tag_list:
                tag_list.remove("Unknown")
            if not tag_list:
                if _line_no in hg_visitor.pattern_matches:
                    hg_visitor.source_code_tags[_line_no]["dl_pipeline_tag"].extend(
                        hg_visitor.pattern_matches[_line_no]["dl_pipeline_tag"]
                    )

        return_info = {
            "analysis_info": analysis_info,
            "source_code_tags": hg_visitor.source_code_tags,
            "cell_callsites": utils.get_line_numbers_cleaned(
                analysis_info["pycg_output"], str(nb_path), file_name
            ),
        }

    return return_info


# Direct run
if __name__ == "__main__":
    pass
