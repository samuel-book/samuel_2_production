import numpy as np
import pandas as pd
import pickle
import shap


from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class ThrombolysisChoiceModel:
    """
    Predicts outcome

    Attributes:

        

    Methods:
        
    """
    
    def __init__(self):
        """
        Load data for modelling, split into X and y, and one-hot encode stroke
        team.
        """

        # Load data
        data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)
        
        # Get X and y
        X_fields = [
            'stroke_team',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'arrival_to_scan_time',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'afib_anticoagulant',
            'age'
        ]

        self.X = data[X_fields]
        self.y = data['thrombolysis']

        # Split 75:25
        strat = data['stroke_team'].map(str) + '-' + data['thrombolysis'].map(
            str)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size = 0.25, stratify=strat, random_state = 42)

        # One hot encode hospitals
        X_train_hosp = pd.get_dummies(self.X_train['stroke_team'], prefix = 'team')
        self.X_train_one_hot = pd.concat([self.X_train, X_train_hosp], axis=1)
        self.X_train_one_hot.drop('stroke_team', axis=1, inplace=True)
        X_test_hosp = pd.get_dummies(self.X_test['stroke_team'], prefix = 'team')
        self.X_test_one_hot = pd.concat([self.X_test, X_test_hosp], axis=1)
        self.X_test_one_hot.drop('stroke_team', axis=1, inplace=True)
        self.X_test.to_csv('./output/thrombolysis_choice_feature_values.csv')


    def get_shap(self):

        # Get SHAP valuess
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values_extended = self.explainer(self.X_test_one_hot)
        self.shap_values = self.shap_values_extended.values
        self.shap_values_df = pd.DataFrame(
            self.shap_values, columns=list(self.X_test_one_hot))

        # Sum hospital SHAPs
        teams = [hosp for hosp in list(self.X_train_one_hot) if hosp[0:4]=='team']
        self.shap_values_df['hospital'] = self.shap_values_df[teams].sum(axis=1)
        for team in teams:
            self.shap_values_df.drop(team, axis=1, inplace=True)

        # Add total SHAP
        self.shap_values_df['total'] = self.shap_values_df.sum(axis=1)

        # Add base value and reorder DataFrame to put base value first
        cols = list(self.shap_values_df)
        self.shap_values_df['base'] = self.shap_values_extended.base_values[0]
        cols.insert(0, 'base')
        self.shap_values_df = self.shap_values_df[cols]

        # Add probability
        odds = np.exp(self.shap_values_df['total'] ) 
        self.shap_values_df['probability'] = odds / (1 + odds)

        # Get average hospital SHAP values
        self.shap_values_df['stroke_team'] = self.X_test['stroke_team'].values
        self.hospital_mean_shap = \
            self.shap_values_df.groupby('stroke_team').mean()['hospital']
        self.hospital_mean_shap.to_csv(
            './output/thrombolysis_choice_hospital_shap.csv')
        self.shap_values_df.drop('stroke_team', axis=1, inplace=True)

        # Save

        self.shap_values_df.to_csv('./output/thrombolysis_choice_shap.csv')


    
    def train_model(self):

        # Define and Fit model
        self.model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        self.model.fit(self.X_train_one_hot, self.y_train)
        # Save using pickle
        filename = './models/thrombolysis_choice.p'
        with open(filename, 'wb') as filehandler:
            pickle.dump(self.model, filehandler)
        # Get predictions for test set
        self.y_pred_proba = self.model.predict_proba(self.X_test_one_hot)[:, 1]
        self.y_pred = self.y_pred_proba >= 0.5
        y_pred_proba_df = pd.DataFrame()
        y_pred_proba_df['probability'] = self.y_pred_proba
        y_pred_proba_df['predicted'] = self.y_pred
        y_pred_proba_df['observed'] = self.y_test

        y_pred_proba_df.to_csv('./output/thrombolysis_choice_test_predictions.csv')
        # Get accuracy of test set
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        print(f'Accuracy: {accuracy:0.3f}')
        balanced_accuracy = metrics.balanced_accuracy_score(self.y_test, self.y_pred)
        print(f'Balanced accuracy: {balanced_accuracy:0.3f}')
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f'ROC AUC: {roc_auc:0.3f}')
        print(f'Actual thrombolysis: {np.mean(self.y_test):0.3f}')
        print(f'Predicted thrombolysis: {np.mean(self.y_pred):0.3f}')
        