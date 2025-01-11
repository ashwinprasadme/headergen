import json
import os
import pathlib
import pickle
from pathlib import Path

import numpy
import pygtrie

from framework_models.ml_function_classifier import MLFunctionClassifier

ml_function_classifier = MLFunctionClassifier.MLFunctionClassifier()
ML_PIPELINE_MODEL = True
CACHE_PREDICTIONS = {}

SCRIPT_ROOT = pathlib.Path(__file__).parent.absolute()
cache_models = True
# cache_models = True

MODELS = {}
PHASES = {
    "IMPORTS": "Imports",
    "CONFIGURATION": "Configuration",
    "FUNCTION_DEFINITION": "Function Definition",
    "DECLARED_FUNCTION": "Declared Function",
    "BUILTIN_FUNCTION": "Builtin Function",
    "LIBRARY_LOADING": "Library Loading",
    "DATA_LOADING": "Data Loading",
    "VISUALIZATION": "Visualization",
    "UNKNOWN": "Unknown",
    "DATA_CLEANING_PREPARATION": "Data Preparation",
    "DATA_PROFILING_AND_EXPLORATORY_DATA_ANALYSIS": "Data Profiling and Exploratory Data Analysis",
    "DATA_CLEANING_FILTERING": "Data Cleaning Filtering",
    "DATA_SUB_SAMPLING_AND_TRAIN_TEST_SPLITTING": "Data Sub-sampling and Train-test Splitting",
    "FEATURE_ENGINEERING": "Feature Engineering",
    "FEATURE_TRANSFORMATION": "Feature Transformation",
    "FEATURE_SELECTION": "Feature Selection",
    "MODEL_BUILDING_AND_TRAINING": "Model Building and Training",
    "MODEL_TRAINING": "Model Training",
    "MODEL_PARAMETER_TUNING": "Model Parameter Tuning",
    "MODEL_VALIDATION_AND_ASSEMBLING": "Model Validation and Assembling",
}

PHASES_EXPLANATION = {
    # "IMPORTS": "Imports",
    # "CONFIGURATION": "Configuration",
    # "FUNCTION_DEFINITION": "Function Definition",
    # "DECLARED_FUNCTION": "Declared Function",
    # "BUILTIN_FUNCTION": "Builtin Function",
    # "LIBRARY_LOADING": "Library Loading",
    "DATA_LOADING": "The process of acquiring data that will be used to build a model",
    "VISUALIZATION": "Visualization of data using plots and graphical techniques",
    # "UNKNOWN": "Unknown",
    # "DATA_CLEANING_PREPARATION": "Data Cleaning Preparation",
    "DATA_PROFILING_AND_EXPLORATORY_DATA_ANALYSIS": "Assess overall data quality, detect noise in labels, skewness and correlations, class imbalance, and target distributions, bias detection",
    "DATA_CLEANING_FILTERING": "Auto Cleaning or filtering, filling missing values, removing outliers, bias mitigation",
    "DATA_SUB_SAMPLING_AND_TRAIN_TEST_SPLITTING": "Determine the best strategy for train-test splitting and sub-sampling methodology for large data",
    # "FEATURE_ENGINEERING": "Feature Engineering",
    "FEATURE_TRANSFORMATION": "Determine a ranked list of features that should be added or dropped, determine the best encoding strategy for certain features automatically, identify column concepts and link them to derive useful new features based on domain knowledge ",
    "FEATURE_SELECTION": "Select the best subset of features",
    # "MODEL_BUILDING_AND_TRAINING": "Model Building and Training",
    "MODEL_TRAINING": "The process of applying the right model to fit to the data and train the model",
    "MODEL_PARAMETER_TUNING": "Hypter parameter choosing and tuning",
    "MODEL_VALIDATION_AND_ASSEMBLING": "The process of ensuring the generalizability of a model, evaluating and comparing models, assembling multiple models",
}

PHASE_GROUPS = {
    "Library Loading": ["Library Loading"],
    "Visualization": ["Visualization"],
    "Others": ["Others"],
    "Data Preparation": [
        "Data Preparation",
        "Data Profiling and Exploratory Data Analysis",
        "Data Preparation and Exploration",
        "Data Cleaning Filtering",
        "Data Sub-sampling and Train-test Splitting",
        "Data Loading",
        "Exploratory Data Analysis",
    ],
    "Feature Engineering": [
        "Feature Engineering",
        "Feature Transformation",
        "Feature Selection",
    ],
    "Model Building and Training": [
        "Model Building and Training",
        "Model Training",
        "Model Parameter Tuning",
        "Model Validation and Assembling",
    ],
}

DASWOW_PHASES = {
    "helper_functions": "Helper Functions",
    "load_data": "Load Data",
    "data_preprocessing": "Data Preprocessing",
    "data_exploration": "Data Exploration",
    "modelling": "Modelling",
    "evaluation": "Evaluation",
    "prediction": "Prediction",
    "result_visualization": "Result Visualization",
    "save_results": "Save Results",
    "comment_only": "Comment Only",
}


def get_high_level_phase(phase):
    for _k, _v in PHASE_GROUPS.items():
        if phase in _v:
            return _k

    return PHASES["UNKNOWN"]


ML_MODULES = {
    "keras": os.path.join(SCRIPT_ROOT, "model", "keras"),
    "lightgbm": os.path.join(SCRIPT_ROOT, "model", "lightgbm"),
    "matplotlib": os.path.join(SCRIPT_ROOT, "model", "matplotlib"),
    "numpy": os.path.join(SCRIPT_ROOT, "model", "numpy"),
    "pandas": os.path.join(SCRIPT_ROOT, "model", "pandas"),
    "plotly": os.path.join(SCRIPT_ROOT, "model", "plotly"),
    "scipy": os.path.join(SCRIPT_ROOT, "model", "scipy"),
    "seaborn": os.path.join(SCRIPT_ROOT, "model", "seaborn"),
    "sklearn": os.path.join(SCRIPT_ROOT, "model", "sklearn"),
    "xgboost": os.path.join(SCRIPT_ROOT, "model", "xgboost"),
    "statsmodels": os.path.join(SCRIPT_ROOT, "model", "statsmodels"),
    "tensorflow": os.path.join(SCRIPT_ROOT, "model", "tensorflow"),
    # "tensorflow": os.path.join(SCRIPT_ROOT, "tensorflow"),
}

ML_MODULES_ALIAS = {
    "keras": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "keras.json")).read()
    ),
    "matplotlib": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "matplotlib.json")).read()
    ),
    "numpy": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "numpy.json")).read()
    ),
    "pandas": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "pandas.json")).read()
    ),
    "tensorflow": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "tensorflow.json")).read()
    ),
    "sklearn": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "sklearn.json")).read()
    ),
    "nibabel": json.loads(
        open(os.path.join(SCRIPT_ROOT, "aliases", "nibabel.json")).read()
    ),  # "tensorflow": os.path.SCRIPT_ROOT, "tensorflow"),
}

MODELS = {k: pygtrie.StringTrie(separator=".") for k in ML_MODULES}


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
            if filepath.parts[-i].endswith(".json"):
                module_string.append(filepath.stem)
            else:
                module_string.append(filepath.parts[-i])

    return ".".join(reversed(module_string))


if cache_models and os.path.exists(os.path.join(SCRIPT_ROOT, "models_cache.pickle")):
    MODELS = pickle.load(open(os.path.join(SCRIPT_ROOT, "models_cache.pickle"), "rb"))
else:
    for _module_name, _module_path in ML_MODULES.items():
        counter = 0
        for _file in Path(_module_path).rglob("*.json"):
            # if "numpy" in _file.name:
            #     print()
            print(counter, _file)
            module_name = get_relative_module_name(_file, _module_name, _module_path)
            MODELS[_module_name][module_name] = json.loads(open(_file).read())
            counter += 1

    if cache_models:
        pickle.dump(
            MODELS, open(os.path.join(SCRIPT_ROOT, "models_cache.pickle"), "wb")
        )


def check_alias(func_call):
    root_module = func_call.split(".")[0]

    if root_module in ML_MODULES_ALIAS:
        if func_call in ML_MODULES_ALIAS[root_module]:
            func_call = ML_MODULES_ALIAS[root_module][func_call]

    return func_call


def lookup_pipeline_tag_builtin(func_call):
    if func_call.startswith("<list>"):
        return [PHASES["DATA_CLEANING_PREPARATION"]]
    elif func_call == ("<builtin>.open"):
        return [PHASES["DATA_CLEANING_PREPARATION"]]
    else:
        return [PHASES["UNKNOWN"]]


def lookup_pipeline_tag(func_call, doc_string=""):
    if ML_PIPELINE_MODEL:
        # Check if its available from cache, else try ml
        if func_call in CACHE_PREDICTIONS:
            return CACHE_PREDICTIONS[func_call]
        else:
            _res = lookup_pipeline_tag_ml(func_call, doc_string)
            CACHE_PREDICTIONS[func_call] = _res
            return _res

    else:
        return lookup_pipeline_tag_hg(func_call)


def lookup_pipeline_tag_hg(func_call):
    root_module = func_call.split(".")[0]
    if root_module not in MODELS:
        return [PHASES["UNKNOWN"]]

    if root_module in ML_MODULES_ALIAS:
        if func_call in ML_MODULES_ALIAS[root_module]:
            func_call = ML_MODULES_ALIAS[root_module][func_call]

    try:
        model_match = MODELS[root_module].longest_prefix(func_call)
        _res = [PHASES[k] for k in model_match.value[func_call]]
        if _res:
            return _res
        else:
            return [PHASES["UNKNOWN"]]
    except:
        # print("ML tag missing for:", func_call)
        return [PHASES["UNKNOWN"]]


def lookup_pipeline_tag_ml(func_call, doc_string):
    root_module = func_call.split(".")[0]

    if root_module in ML_MODULES_ALIAS:
        if func_call in ML_MODULES_ALIAS[root_module]:
            func_call = ML_MODULES_ALIAS[root_module][func_call]

    try:
        _res = ml_function_classifier.predict_function(func_call, doc_string)
        # _classic_res = lookup_pipeline_tag(func_call)

        if not _res:
            _res = [PHASES["UNKNOWN"]]
    except:
        print("ML tag missing for:", func_call)
        return [PHASES["UNKNOWN"]]

    return _res


if __name__ == "__main__":
    for line in open("test.txt").readlines():
        # print(line.strip())
        # print(lookup_pipeline_tag(line.strip()))

        lookup_pipeline_tag(line.strip())
        # lookup_pipeline_tag("keras.backend.get_session")


"""
Pipeline Phases

LIBRARY_LOADING
DATA_LOADING
VISUALIZATION
UNKNOWN
DATA_CLEANING_PREPARATION
DATA_PROFILING_AND_EXPLORATORY_DATA_ANALYSIS
DATA_CLEANING_FILTERING
DATA_SUB_SAMPLING_AND_TRAIN_TEST_SPLITTING
FEATURE_ENGINEERING
FEATURE_TRANSFORMATION
FEATURE_SELECTION
MODEL_BUILDING_AND_TRAINING
MODEL_TRAINING
MODEL_PARAMETER_TUNING
MODEL_VALIDATION_AND_ASSEMBLING
"""

# MODEL_SAVING_AND_LOADING
