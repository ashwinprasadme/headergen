import logging
import os

import aiofiles
import nbformat
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from framework_models import get_high_level_phase, DASWOW_PHASES
from headergen import headergen

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Enable GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/get_analysis_notebook/")
async def get_analysis(file: UploadFile = File(...)):
    """Upload a notebook file, analyze chunks of it, and add metadata."""
    try:
        # Save the uploaded file to the uploads directory
        file_location = f"{UPLOAD_DIR}/{file.filename}"

        async with aiofiles.open(file_location, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Load the notebook
        async with aiofiles.open(file_location, "r", encoding="utf-8") as file:
            notebook_content = await file.read()
            notebook = nbformat.reads(notebook_content, as_version=4)

        # Perform analysis on the uploaded notebook
        try:
            analysis_meta = headergen.start_headergen(
                file_location, OUTPUT_DIR, debug_mode=True
            )
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

        # Prepare the analysis output in a chunked dictionary, mapping analysis to cells
        analysis_output = {"cell_mapping": {}}

        if "block_mapping" in analysis_meta:
            for cell_index, cell_results in analysis_meta["block_mapping"].items():
                # Get high-level phases and convert set to list
                ml_phases = list(set([DASWOW_PHASES.get(tag, "Unknown") for tag in cell_results["dl_pipeline_tag"]]))
                func_list = {k:{"doc_string":v, "arguments":[]} for k,v in cell_results.get("doc_string", {}).items()}

                for call_args in cell_results["call_args"].values():
                    for call, args in call_args.items():
                        if call in func_list:
                            func_list[call]["arguments"].append(args)

                # Add to the chunked dictionary without modifying the content
                analysis_output["cell_mapping"][cell_index] = {
                    "ml_phase": ml_phases,  # Ensure ml_phases is a list, not a set
                    "functions": func_list,
                }

        # Return the chunked analysis output without overwriting notebook content
        return JSONResponse(content=analysis_output)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000)

# Example curl request to use get_analysis_notebook with a file path
# Replace 'your_notebook.ipynb' with the path to your actual notebook file
# curl -X POST "http://localhost:8000/get_analysis_notebook/" \
#     -H "accept: application/json" \
#     -H "Content-Type: multipart/form-data" \
#     -F "file=@/mnt/Projects/PhD/Research/HeaderGen/git_sources/headergen_githib/.scrapy/notebooks/01-keras-deep-learning-to-solve-titanic.ipynb"
