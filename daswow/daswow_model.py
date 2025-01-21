# %%
import os
from string import punctuation

import joblib
import numpy as np
from nltk.corpus import stopwords

from daswow.CellFeatures import CellFeatures
from daswow.model_download import download_models_from_github_release

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

download_models_from_github_release()

class Preprocessing:
    # init. set dataframe to be processed
    def __init__(self, df):
        self.df = df
        self.features = ["text"]
        self.stopWords = set(stopwords.words("english"))

    def remove_stopwords(self, words):
        wordsFiltered = [w for w in words if w not in self.stopWords]
        return wordsFiltered

    def set_column(self, col, newcol):
        self.df[newcol] = self.df[col].apply(self.combine_lists_to_text)
        return self.df

    def custom_text_preprocessing(self, s):
        favourite_punc = [".", "#", "_"]
        if s:
            for char in punctuation:
                if char not in favourite_punc:
                    s = s.replace(char, " ")
            s = " ".join(
                [
                    "" if word.replace(".", "").isdigit() else word
                    for word in s.split(" ")
                ]
            )
            # s = " ".join(['$' if '$' in word and word.replace('$','').isnumeric() else word for word in s.split(' ')])
            s = " ".join(self.remove_stopwords(s.lower().split(" ")))
            s = " ".join([word.strip() for word in s.split(" ") if len(word) > 1])
            # s = " ".join([word for word in s if word not in throw_words])
        return s

    def combine_lists_to_text(self, obj):
        text = ""
        if obj:
            try:
                if isinstance(obj, list):
                    for element in obj:
                        if isinstance(element, list):
                            for e in element:
                                text = text + " " + str(e)
                        else:
                            text = text + " " + str(element)
                elif isinstance(obj, str):
                    text = text + " " + obj
            except:
                print("expecting string or list, found %s" % type(obj))

            text = text.strip().lower()
        return text

    def set_lexical(self, features):
        new_text = []
        for idx, row in self.df.iterrows():
            l = []
            for each in features:
                if isinstance(row[each], list):
                    l = l + row[each]
                else:
                    l = l + [row[each]]
            new_text.append(l)
        self.df["new_text"] = new_text
        return self.df

    def process(self):
        self.df = self.set_lexical(self.features)
        self.df["new_text"] = self.df["text"].apply(self.combine_lists_to_text)
        self.df["new_text"] = self.df["new_text"].apply(self.custom_text_preprocessing)
        return self.df


class DASWOWInference:
    def __init__(self, nb_path, models_path=MODELS_PATH):
        cf = CellFeatures()
        self.df = cf.get_cell_features_nb(nb_path)

        self.preprocesser = Preprocessing(self.df)
        self.model = joblib.load(f"{models_path}/rf_code_scaled.pkl")
        self.tfidf = joblib.load(f"{models_path}/tfidf_vectorizer.pkl")
        self.selector = joblib.load(f"{models_path}/selector.pkl")
        self.ss = joblib.load(f"{models_path}/scaler.pkl")
        self.stopWords = set(stopwords.words("english"))
        self.stat_features = [
            "linesofcomment",
            "linesofcode",
            "variable_count",
            "function_count",
        ]
        self.labels = [
            "helper_functions",
            "load_data",
            "data_preprocessing",
            "data_exploration",
            "modelling",
            "evaluation",
            "prediction",
            "result_visualization",
            "save_results",
            "comment_only",
        ]

    def remove_stopwords(self, words):
        wordsFiltered = [w for w in words if w not in self.stopWords]
        return wordsFiltered

    def preprocess(self):
        self.df = self.preprocesser.process()
        return True

    def vectorize(self):
        text = self.tfidf.transform(self.df["new_text"])
        return text

    def select_features(self, text):
        text = self.selector.transform(text)
        return text

    def set_statistical_features(self, text):
        X_copy = text.toarray()

        for each in self.stat_features:
            X_copy = np.c_[X_copy, self.df[each].values]

        return X_copy

    def scale_features(self, text):
        text = self.ss.transform(text)
        return text

    def predict(self):
        self.preprocess()
        cells_features = self.vectorize()
        cells_features = self.select_features(cells_features)
        cells_features = self.set_statistical_features(cells_features)
        cells_features = self.scale_features(cells_features)
        prediction = self.model.predict(cells_features)
        # convert prediction to labels
        prediction = [
            [self.labels[i] for i, p in enumerate(pred) if p == 1]
            for pred in prediction
        ]
        return prediction


if __name__ == "__main__":
    nb_path = "/mnt/Projects/PhD/Research/Student-Thesis/7_Akshita/daswow-data-science-code-analysis/.scrapy/user_study_notebooks/user_study_notebooks/cyclegan-with-data-augmentation.ipynb"
    infer = DASWOWInference(
        nb_path=nb_path,
    )

    infer.predict()
