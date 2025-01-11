from pathlib import Path

import click
import simplejson as sjson
import uvicorn

from framework_models import get_high_level_phase
from headergen import headergen
from headergen.server import app


@click.group()
def cli():
    """CLI tool"""
    pass


@cli.command()
@click.option(
    "-i", "--input", required=True, help="Path of the input Notebook/Python script"
)
@click.option("-o", "--output", default=".", help="Output path")
@click.option(
    "-j", "--json_output", is_flag=True, default=False, help="Output JSON metadata"
)
@click.option(
    "-d",
    "--debug_mode",
    is_flag=True,
    default=False,
    help="Debug mode, keep py files",
)
def generate(input, output, json_output, debug_mode):
    """Generate HeaderGen linted notebook"""
    input = Path(input)
    output = Path(output)

    if not output.is_dir():
        print(f"{output} is not a directory.")
        return

    analysis_meta = headergen.start_headergen(input, output, debug_mode=debug_mode)

    if json_output:
        analysis_output = {
            "cell_callsites": analysis_meta["cell_callsites"],
            "block_mapping": {},
            "analysis_info": analysis_meta["analysis_info"],
        }

        if "block_mapping" in analysis_meta:
            for _cell, _cell_results in analysis_meta["block_mapping"].items():
                analysis_output["block_mapping"][_cell] = list(
                    set(
                        [
                            get_high_level_phase(x)
                            for x in _cell_results["dl_pipeline_tag_counter"]
                            if x
                            not in [
                                "Unknown",
                                "Function Definition",
                                "Builtin Function",
                            ]
                        ]
                    )
                )

        with open(output / (input.stem + ".json"), "w") as f:
            sjson.dump(
                analysis_output,
                f,
                indent=4,
                iterable_as_array=True,
            )


@cli.command()
@click.option(
    "-i", "--input", required=True, help="Path of the input Notebook/Python script"
)
@click.option("-o", "--output", default=".", help="Output path")
@click.option(
    "-j", "--json_output", is_flag=True, default=False, help="Output JSON metadata"
)
def types(input, output, json_output):
    """Generate Type information"""
    input = Path(input)
    output = Path(output)

    if not output.is_dir():
        print(f"{output} is not a directory.")
        return

    analysis_meta = headergen.get_analysis_output(input, output)

    print(
        sjson.dumps(
            analysis_meta["types_formatted"],
            indent=4,
            iterable_as_array=True,
        )
    )

    if json_output:
        with open(output / (input.stem + "_types.json"), "w") as f:
            sjson.dump(
                analysis_meta["types_formatted"],
                f,
                indent=4,
                iterable_as_array=True,
            )


@cli.command()
def server():
    """Start Server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)
