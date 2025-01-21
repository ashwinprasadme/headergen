# %%
import errno
import logging
import os
import shutil
import time
import traceback
from pathlib import Path

import simplejson as sjson

from headergen import headergen

script_dir = os.path.abspath(os.path.dirname(__file__))

# Careful, the out_path folder will be removed
file_path = f"/mnt/Projects/PhD/Research/HeaderGen/git_sources/headergen_githib/.scrapy/notebooks/01-keras-deep-learning-to-solve-titanic.ipynb"
out_path = f"{script_dir}/results/"


# Remove results dir and recreate
try:
    shutil.rmtree(out_path)
except OSError:
    print("Removal of the directory %s failed" % out_path)

try:
    os.mkdir(out_path)
except OSError:
    print("Creation of the directory %s failed" % out_path)
else:
    print("Successfully created the directory %s " % out_path)

SAVE_HEADERGEN_OUTPUT = True

# Logging
logging.basicConfig(
    filename="{}/logs.log".format(out_path),
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=3,
)

start = time.time()

analysis_meta = headergen.start_headergen(file_path, out_path, debug_mode=True)
# analysis_meta = headergen.get_analysis_output(file_path, out_path)

print(f"Analysis took: {time.time() - start}")

with open("{}/callsites.json".format(out_path), "w") as outfile:
    sjson.dump(
        dict(sorted(analysis_meta["cell_callsites"].items(), key=lambda e: int(e[0]))),
        outfile,
        indent=4,
        iterable_as_array=True,
    )

with open("{}/metadata.json".format(out_path), "w") as outfile:
    sjson.dump(
        analysis_meta,
        outfile,
        indent=4,
        iterable_as_array=True,
    )

logging.debug("DONE")
