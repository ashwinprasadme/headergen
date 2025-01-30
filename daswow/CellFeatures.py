# coding: utf-8

"""
This class implements necessary methods for extracting features from notebooks. Most of the methods are self-explanatory. Comments are available wherever necessary.
"""

import json

# import statements
import math
import os
import re
import string
from collections import namedtuple
from glob import glob
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd


class CellFeatures:
    # init function, set path (where notebook is) and files (list of notebooks) for which features are to be extracted
    # store path indicates where extracted feature pickles are stored (separately for each notebook)
    def __init__(self, path=None, files=None, store_path=None):
        self.path = path
        self.files = files
        self.df = pd.DataFrame()
        # pickles path to store pickles
        self.store_path = store_path

    def get_help(self):
        help_text = """This class (NotebookParser) implements necessary methods for extracting features from notebooks.
        Class is initialized with params: path_to_folder_containing_notebooks, list_of_notebooks, path_to_store_pickles.
        get_cell_features() is the main method for extracting features.
        A dataframe containing all features (all notebooks) is returned.
        Each row in the dataframe indicates a cell in a notebook.
        A .pkl for each notebook separetly is stored in store_path specified.
        Happy machine learning!"""
        return help_text

    # if files are not mentioned, features for all notebooks in the given path are generated.
    def get_files(self):
        if len(self.files) < 1:
            pfiles = glob(self.path + "\*.ipynb")
            # get file names
            for pfile in pfiles:
                self.files.append(pfile.split(os.path.sep)[-1])
        if len(self.files) < 1:
            print("No .ipynb files found")

    # json reading of notebook
    def read_file(self, text):
        data = ""
        try:
            data = json.loads(text)
            # print json dict keys if necessary to the type of contents in json formatted jupyter notebook
        except:
            print("Bad json!")
            data = None
        return data

    # variable counter for the given data
    def variable_counter(self, data):
        variable_count = 0
        # if only alphanumerics before '=' then throw the left side
        pattern = re.compile("[\w+.*\s*]=")  # allows underscore,alphanumerics
        for d in data:
            found = pattern.findall(d)
            # print(d, pattern,d.split('=',1)[0])
            if found:
                variable_count = variable_count + 1
        return variable_count

    def function_counter(self, data):
        function_count = 0
        for d in data:
            fwords = [w.strip() for w in d.split(" ")]
            if "def" in fwords:
                function_count = function_count + 1
        return function_count

    def code_processing(self, data):
        # print("***************")
        variable_count = self.variable_counter(data)
        function_count = self.function_counter(data)
        # words_without_variables = variable_remover(data) #can be used if needed, but we do not find any usefulness for this thesis
        # words_without_params = remove_params(data)
        return variable_count, function_count

    def get_import_keywords(self, language):
        if language == "python":
            import_keywords = ["import", "from"]
        elif language == "R":
            import_keywords = ["library", "require"]
        elif language in ["Julia", "julia"]:
            import_keywords = ["import", "importall", "using"]
        else:
            import_keywords = []
        return import_keywords

    def init_values(self):
        return None, None, None, None, None, None, None

    def output_processing(self, data):
        if len(data) <= 1:
            for d in data:
                try:
                    if d["name"]:
                        output_name = d["name"]
                except:
                    output_name = None
                try:
                    if d["output_type"]:
                        output_type = d["output_type"]
                except:
                    output_type = None
                try:
                    output_text = []
                    try:
                        if d["text"]:
                            output_text = list(d["data"].keys())
                    except:
                        pass
                    try:
                        if d["data"]:
                            output_text = list(d["data"].keys())
                    except:
                        pass
                except:
                    output_text = []
        else:
            output_type = ""
            output_name = ""
            output_text = []
            for d in data:
                try:
                    if d["output_type"]:
                        output_type = output_type + d["output_type"] + " "
                except:
                    pass
                try:
                    if d["data"]:
                        output_text.append(list(d["data"].keys()))
                except:
                    pass
                try:
                    if d["text"]:
                        output_text.append(d["text"])
                except:
                    pass
                try:
                    if d["output_name"]:
                        output_name = output_name + d["output_name"] + " "
                except:
                    pass
            if output_type == "":
                output_type = None
            if output_text == []:
                output_text = []
            if output_name == "":
                output_name = None
        return output_name, output_type, output_text

    # function to retrieve all the textual and statistical features from a notebook. Data is notebook.
    def extract_data(self, data, filename):
        df_result = pd.DataFrame()

        if isinstance(data, dict):
            keys = data.keys()
        else:
            keys = []

        # metadata retreival
        if "metadata" in keys:
            metadata_keys = data["metadata"].keys()
        else:
            metadata_keys = []
        # print("metadata_keys: ",metadata_keys)

        # language info retreival
        language = None
        language_version = None
        if "language_info" in metadata_keys:
            language = data["metadata"]["language_info"]["name"]
            language_version = data["metadata"]["language_info"].get("version", None)
        elif "language" in metadata_keys:
            language = data["metadata"]["language"]
            language_version = data["metadata"]["language"]
        else:
            language = None
            language_version = None

        kernel_language = None
        try:
            kernel_language = data["metadata"]["kernelspec"]["name"]
        except:
            kernel_language = None

        # keep track of the immediate markdown before (only first line)
        markdown_heading = None
        # keep track of the immediate code line before (only last line)
        code_line_before = None

        # cell info retreival
        if "cells" in keys:
            cells = data["cells"]
            for cell in cells:
                obj = None
                (
                    comments,
                    variable_count,
                    function_count,
                    execution_count,
                    output_name,
                    output_type,
                    output_text,
                ) = self.init_values()
                # print(cell['metadata']) #not very useful, it is already given by outputs cell

                # common cell elements
                try:
                    if cell["execution_count"]:
                        execution_count = cell["execution_count"]
                    else:
                        execution_count = None
                except:
                    execution_count = None
                try:
                    if cell["outputs"]:
                        output_name, output_type, output_text = self.output_processing(
                            cell["outputs"]
                        )
                    else:
                        output_name = None
                        output_type = None
                        output_text = []
                except:
                    output_name = None
                    output_type = None
                    output_text = []

                # code elements
                if cell["cell_type"] == "code":
                    code, comment, import_text = [], [], []
                    try:
                        if cell["source"]:
                            source_code = None
                            if isinstance(cell["source"], str):
                                source_code = cell["source"].split("\n")
                            elif isinstance(cell["source"], list):
                                source_code = cell["source"]
                            else:
                                source_code = cell["source"]
                            import_keywords = self.get_import_keywords(language)
                            for line in source_code:
                                if line != "\n":  # ignore line spaces
                                    result = line.strip()
                                    if len(result) > 0:
                                        if result[0] == "#":
                                            comment.append(result)
                                        else:
                                            code.append(result)
                                            if result.split(" ")[0] in import_keywords:
                                                r = result.split(" ")
                                                import_text.append(
                                                    " ".join(
                                                        [
                                                            word
                                                            for word in r
                                                            if word
                                                            not in import_keywords
                                                        ]
                                                    )
                                                )

                            if len(code) >= 1:
                                variable_count, function_count = self.code_processing(
                                    code
                                )
                            else:
                                variable_count = 0
                                function_count = 0
                            obj = pd.Series(
                                {
                                    "filename": filename,
                                    "cell_type": "code",
                                    "language": language,
                                    "language_version": language_version,
                                    "kernel_language": kernel_language,
                                    "markdown_heading": markdown_heading,
                                    "text": code,
                                    "comment": comment,
                                    "code_line_before": code_line_before,
                                    "variable_count": variable_count,
                                    "function_count": function_count,
                                    "linesofcode": len(code),
                                    "linesofcomment": len(comment),
                                    "linesofmarkdown": 0,
                                    "execution_count": execution_count,
                                    "output_name": output_name,
                                    "output_type": output_type,
                                    "output_text": output_text,
                                    "import_text": import_text,
                                }
                            )

                    except:
                        obj = pd.Series(
                            {
                                "filename": filename,
                                "cell_type": "code",
                                "language": language,
                                "language_version": language_version,
                                "kernel_language": kernel_language,
                                "markdown_heading": markdown_heading,
                                "text": code,
                                "comment": comment,
                                "code_line_before": code_line_before,
                                "variable_count": variable_count,
                                "function_count": function_count,
                                "linesofcode": len(code),
                                "linesofcomment": len(comment),
                                "linesofmarkdown": 0,
                                "execution_count": execution_count,
                                "output_name": output_name,
                                "output_type": output_type,
                                "output_text": output_text,
                                "import_text": import_text,
                            }
                        )
                        pass
                    try:
                        code_line_before = [code[len(code) - 1]]
                    except:
                        pass
                # markdown elements
                elif cell["cell_type"] == "markdown":
                    markdown = []
                    try:
                        if cell["source"]:  # check for empty cells
                            source_code = None
                            if isinstance(cell["source"], str):
                                source_code = cell["source"].split("\n")
                            elif isinstance(cell["source"], list):
                                source_code = cell["source"]
                            else:
                                source_code = cell["source"]
                            for line in source_code:
                                if line != "\n":  # ignore line spaces
                                    result = line.strip()
                                    if len(result) > 0:
                                        markdown.append(result)
                            # whenever there is a new markdown available, use the first markdown line as the heading for the rest of the cells followed
                            try:
                                markdown_heading = [markdown[0]]
                            except:
                                pass
                            obj = pd.Series(
                                {
                                    "filename": filename,
                                    "cell_type": "markdown",
                                    "language": language,
                                    "language_version": language_version,
                                    "kernel_language": kernel_language,
                                    "markdown_heading": markdown_heading,
                                    "text": markdown,
                                    "comment": None,
                                    "code_line_before": code_line_before,
                                    "variable_count": 0,
                                    "function_count": 0,
                                    "linesofcode": 0,
                                    "linesofcomment": 0,
                                    "linesofmarkdown": len(markdown),
                                    "execution_count": execution_count,
                                    "output_name": output_name,
                                    "output_type": output_type,
                                    "output_text": output_text,
                                    "import_text": None,
                                }
                            )
                    except:
                        obj = pd.Series(
                            {
                                "filename": filename,
                                "cell_type": "markdown",
                                "language": language,
                                "language_version": language_version,
                                "kernel_language": kernel_language,
                                "markdown_heading": markdown_heading,
                                "text": markdown,
                                "comment": None,
                                "code_line_before": code_line_before,
                                "variable_count": 0,
                                "function_count": 0,
                                "linesofcode": 0,
                                "linesofcomment": 0,
                                "linesofmarkdown": len(markdown),
                                "execution_count": execution_count,
                                "output_name": output_name,
                                "output_type": output_type,
                                "output_text": output_text,
                                "import_text": None,
                            }
                        )
                        pass

                else:
                    content = []
                    try:
                        if cell["source"]:  # check for empty cells
                            source_code = None
                            if isinstance(cell["source"], str):
                                source_code = cell["source"].split("\n")
                            elif isinstance(cell["source"], list):
                                source_code = cell["source"]
                            else:
                                source_code = cell["source"]
                            for line in source_code:
                                if line != "\n":  # ignore line spaces
                                    result = line.strip()
                                    if len(result) > 0:
                                        content.append(result)
                    except:
                        pass
                    obj = pd.Series(
                        {
                            "filename": filename,
                            "cell_type": cell["cell_type"],
                            "language": language,
                            "language_version": language_version,
                            "kernel_language": kernel_language,
                            "markdown_heading": markdown_heading,
                            "text": content,
                            "comment": None,
                            "code_line_before": code_line_before,
                            "variable_count": 0,
                            "function_count": 0,
                            "linesofcode": 0,
                            "linesofcomment": 0,
                            "linesofmarkdown": 0,
                            "execution_count": execution_count,
                            "output_name": output_name,
                            "output_type": output_type,
                            "output_text": output_text,
                            "import_text": None,
                        }
                    )

                try:
                    df_result = pd.concat(
                        [df_result, obj.to_frame().T], ignore_index=True
                    )
                    # if obj.iloc[0:].isnull().all():
                    # we still accept this as valid one, since it still accounts for a cell (we need this to keep cell number correct)

                except:
                    pass
                    # if obj is None:

        return df_result

    # main function to call for generating features.
    def get_cell_features(self):
        self.get_files()
        # get file content
        for file in self.files:
            try:
                # read the notebook
                with open(os.path.join(self.path, file), "r", encoding="utf-8") as f:
                    data = self.read_file(f.read())
                try:
                    # if data is available, extract features
                    if data:
                        # print(file)
                        temp_df = self.extract_data(data, file)
                        temp_df["cell_number"] = temp_df.index
                        # add the code_line_after feature once all the cell details are available
                        code_line_after = []
                        for idx, row in temp_df.iterrows():
                            try:
                                # get next code cell and get its first line
                                indices = list(temp_df["cell_type"].values[idx + 1 :])
                                first_find = indices.index("code")
                                code_line_after.append(
                                    [temp_df.iloc[first_find + idx + 1]["text"][0]]
                                )
                            except:
                                code_line_after.append(None)
                        # once features are loaded, store it in pickle
                        temp_df["code_line_after"] = code_line_after
                        temp_df.to_pickle(self.store_path + file + ".pkl")
                        print("storing pickle in", self.store_path + file + ".pkl")
                        self.df = self.df.append(temp_df)
                except:
                    pass
                    print("unable to extract data from the file ", file)
            except:
                pass
                print("File not there", file)
        # set index and return the combined dataframe (all notebooks)
        self.df.index = range(self.df.shape[0])
        return self.df

    # main function to call for generating features.
    def get_cell_features_nb(self, nb_path):
        try:
            # read the notebook
            with open(os.path.join(nb_path), "r", encoding="utf-8") as f:
                data = self.read_file(f.read())
            try:
                file_name = nb_path.split(os.path.sep)[-1]
                # if data is available, extract features
                if data:
                    # print(file)
                    temp_df = self.extract_data(data, file_name)
                    temp_df["cell_number"] = temp_df.index
                    # add the code_line_after feature once all the cell details are available
                    code_line_after = []
                    for idx, row in temp_df.iterrows():
                        try:
                            # get next code cell and get its first line
                            indices = list(temp_df["cell_type"].values[idx + 1 :])
                            first_find = indices.index("code")
                            code_line_after.append(
                                [temp_df.iloc[first_find + idx + 1]["text"][0]]
                            )
                        except:
                            code_line_after.append(None)
                    # once features are loaded, store it in pickle
                    temp_df["code_line_after"] = code_line_after
                    temp_df.to_pickle(self.store_path + file_name + ".pkl")
                    print("storing pickle in", self.store_path + file_name + ".pkl")
            except Exception as e:
                print("unable to extract data from the file ", file_name)
                print(e)
        except:
            print("File not there", file_name)

        return temp_df
