import os
from download_from_ghub import download_file_from_github_release
import pickle


class CellClassifier:
    def __init__(self):        
        # if the folder does not exist then create the folder
        if not os.path.exists('cell_classifier'):
            os.mkdir('cell_classifier')
        
        # if the file does not exist then download the model from a github release
        if not os.path.exists('cell_classifier/code_classifier.pkl'):
            # download the model from github release
            download_file_from_github_release()
        
        # load the model from the pickle file
        with open('cell_classifier/code_classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        
        # classes of the model
        self.classes = ['helper_functions','load_data','data_preprocessing',
               'data_exploration','modelling','evaluation',
               'prediction','result_visualization','save_results',
               'comment_only']
    
    # function to predict the workflow step of a code cell, given the code cell as input, returns the predicted workflow step, e.g., 'load_data'
    def predict_workflow_step(self, codestring):
        return self.classes[self.classifier.predict(codestring)]