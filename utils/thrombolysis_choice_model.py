import numpy as np
import pandas as pd
import pickle
import shap

from xgboost import XGBClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn import metrics

class ThrombolysisChoiceModel:
    """
    Predicts outcome

    Attributes:

        

    Methods:
        
    """
    
    def __init__(self):
        """
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

        # Split 75:25 (split in order to align with outcome model)
        length = len(self.X)
        train_length = int(length * 0.75)
        self.X_train = self.X.iloc[0:train_length]
        self.y_train = self.y.iloc[0:train_length]
        self.X_test = self.X.iloc[train_length:]
        self.y_test = self.y.iloc[train_length:]

        # One hot encode hospitals
        X_train_hosp = pd.get_dummies(self.X_train['stroke_team'], prefix = 'team')
        self.X_train_one_hot = pd.concat([self.X_train, X_train_hosp], axis=1)
        self.X_train_one_hot.drop('stroke_team', axis=1, inplace=True)
        X_test_hosp = pd.get_dummies(self.X_test['stroke_team'], prefix = 'team')
        self.X_test_one_hot = pd.concat([self.X_test, X_test_hosp], axis=1)
        self.X_test_one_hot.drop('stroke_team', axis=1, inplace=True)

    def get_shap(self):
        # Load model
        filename = './models/thrombolysis_choice.p'
        with open(filename, 'rb') as filehandler:
            model = pickle.load(filehandler)
        # Get SHAP values
        self.explainer = shap.TreeExplainer(model)
        self.shap_values_extended = self.explainer(self.X_test_one_hot)
        self.shap_values = self.shap_values_extended.values
        self.shap_values_df = pd.DataFrame(
            self.shap_values, columns=list(self.X_train_one_hot))
        
        # Sum hopsital SHAPs
        teams = [hosp for hosp in list(self.X_train_one_hot) if hosp[0:4]=='team']
        self.shap_values_df['hopsital'] = self.shap_values_df[teams].sum(axis=1)
        for team in teams:
            self.shap_values_df.drop(team, axis=1, inplace=True)   
    
    def train_model(self):

        # Define and Fit model
        model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        model.fit(self.X_train_one_hot, self.y_train)
        # Save using pickle
        filename = './models/thrombolysis_choice.p'
        with open(filename, 'wb') as filehandler:
            pickle.dump(model, filehandler)
        # Get predictions for test set
        y_pred_proba = model.predict_proba(self.X_test_one_hot)[:, 1]
        y_pred = y_pred_proba >= 0.5
        y_pred_proba_df = pd.DataFrame()
        y_pred_proba_df['probability'] = y_pred_proba
        y_pred_proba_df.to_csv('./output/thrombolysis_choice_test_predictions.csv')
        # Get accuracy of test set
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy:0.3f}')
        balanced_accuracy = metrics.balanced_accuracy_score(self.y_test, y_pred)
        print(f'Balanced accuracy: {balanced_accuracy:0.3f}')
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f'ROC AUC: {roc_auc:0.3f}')
        print(f'Actual thrombolysis: {np.mean(self.y_test):0.3f}')
        print(f'Predicted thrombolysis: {np.mean(y_pred):0.3f}')



        




