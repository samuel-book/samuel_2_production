import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class OutcomeModel():
    """
    Predicts outcome

    Attributes:



    Methods:

    """

    def __init__(self):
        """
        """

        # Load data
        self.data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)
        
        self.prototype_patients = pd.read_csv(
            './data/data_for_ml/ml_patient_prototypes_for_outcomes.csv',
            index_col='Patient prototype')
        
        # Get X and y
        self.X_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis',
            'infarction']

        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()

        self.X = self.data[self.X_fields]
        self.y = self.data['discharge_disability']
        self.prototype_patients = self.prototype_patients[self.X_fields]


        # Split 75:25
        strat = self.data['discharge_disability'].values
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.25, stratify=strat, random_state=42)

        # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(self.X_train[['stroke_team']])
        one_hot_encoded = encoder.transform(self.X_train[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=self.X_train.index)
        self.X_train_one_hot = pd.concat([self.X_train, one_hot_encoded_df], axis=1)
        self.X_train_one_hot.drop('stroke_team', axis=1, inplace=True)
        one_hot_encoded = encoder.transform(self.X_test[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=self.X_test.index)
        self.X_test_one_hot = pd.concat([self.X_test, one_hot_encoded_df], axis=1)
        self.X_test_one_hot.drop('stroke_team', axis=1, inplace=True)

        self.X_test.to_csv('./output/thrombolysis_outcome_feature_values.csv')


