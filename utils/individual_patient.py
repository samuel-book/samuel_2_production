import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
import scipy.stats
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

class IndividualPatientModel:

    """
    Class for individual patient models (thrombolysis choice and outcome).
    """

    def __init__(self):
        """
        Initialize the class.
        """

        self.thrombolysis_choice_fields = [
            'stroke_team',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'arrival_to_scan_time',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'afib_anticoagulant',
            'age',
            'thrombolysis'
        ]

        self.thrombolysis_outcome_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis',
            'discharge_disability'
        ]

        self.data = pd.read_csv(
            './data/data_for_ml/complete_ml_data.csv', low_memory=False)
        
        # Set up one hot encoder
        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()
        enc = OneHotEncoder(categories=[self.stroke_teams])
        
        # Get thrombolysis data
        thrombolysis_data = self.data[self.thrombolysis_choice_fields]
        one_hot = enc.fit_transform(thrombolysis_data[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        thrombolysis_data = pd.concat([thrombolysis_data, one_hot], axis=1)
        self.thrombolysis_data = thrombolysis_data.drop(columns=['stroke_team'])

        # Get stroke outcome data       
        outcome_data = self.data.copy()
        # Restrict fields
        outcome_data = outcome_data[self.thrombolysis_outcome_fields]
        # Remove patients with missing outcome data
        mask = outcome_data['discharge_disability'].isna() == False
        outcome_data = outcome_data[mask]
        # One hot encode stroke teams
        one_hot = enc.fit_transform(outcome_data[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        outcome_data = pd.concat([outcome_data, one_hot], axis=1)
        self.outcome_data = outcome_data.drop(columns=['stroke_team'])
        self.outcome_data['discharge_disability'] = \
            self.outcome_data['discharge_disability'].astype(int)



    def train_models(self, replicates=100):
        """
        Train and save the models.
        """
        
        # THROMBOLYSIS CHOICE MODELS
        print('Training thrombolysis choice models...')

        # Fit models
        model_full = []
        for i in range(replicates):
            # Sample data
            sample = self.thrombolysis_data.sample(frac=1.0, random_state=42+i, replace=True)
            X = sample.drop(columns=['thrombolysis'])
            y = sample['thrombolysis']
            # Fit full model
            model = XGBClassifier(random_state=42+i, learning_rate=0.5)
            model.fit(X, y)
            model_full.append(model)
        # Pickle models
        pickle.dump(model_full, open('./pickled_models/replicate_choice_models.pkl', 'wb'))

        # THROMBOLYSIS OUTCOME MODELS
        print('Training thrombolysis outcome models...')
        # Get relevant data
        

        # Fit models
        model_full = []
        for i in range(replicates):
            # Sample data
            sample = self.outcome_data.sample(frac=1.0, random_state=42+i, replace=True)
            X = sample.drop(columns=['discharge_disability'])
            y = sample['discharge_disability'].values
            y = y.astype(int)
            # Fit full model
            model = XGBClassifier(random_state=42+i)
            model.fit(X, y)
            model_full.append(model)
        # Pickle models
        pickle.dump(model_full, open('./pickled_models/replicate_outcome_models.pkl', 'wb'))





