import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
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

        # One hot encode prototype patients
        one_hot_encoded = encoder.transform(self.prototype_patients[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=self.prototype_patients.index)
        self.prototype_patients_one_hot = pd.concat([self.prototype_patients, one_hot_encoded_df], axis=1)
        self.prototype_patients_one_hot.drop('stroke_team', axis=1, inplace=True)


        # save
        self.X_test.to_csv('./output/thrombolysis_outcome_feature_values.csv')


    def predict_prototype_patients(self):
        """
        Predict outcomes for prototype patients with and without thrombolysis
        """
        self.prototype_patients_outcomes_treated = \
            self.model.predict_proba(self.prototype_patients_one_hot)
        
        untreated_patients  = self.prototype_patients_one_hot.copy()
        untreated_patients['onset_to_thrombolysis'] = -10
        untreated_patients['onset_to_thrombolysis']
        self.prototype_patients_outcomes_untreated = \
            self.model.predict_proba(untreated_patients)
        

    
    def run(self):
        """
        Train model, get SHAP values, and estimate benchmark thrombolysis rates
        """

        self.train_model()
        self.test_model()
        self.predict_prototype_patients()

    
    def test_model(self):

        """
        Test model
        """

        # Get predicted probabilities
        y_probs = self.model.predict_proba(self.X_test_one_hot)
        y_pred = self.model.predict(self.X_test_one_hot)

        # Plot confusion matrix
        cm = confusion_matrix(self.y_test.astype(np.int8), y_pred)
        fig = plt.figure(figsize=(6,6))
        ax1 = fig.add_subplot(1,1,1)
        heatmap = plt.imshow(cm, cmap=plt.cm.Blues)
        colorbar = plt.colorbar(heatmap, shrink=0.8, ax=ax1, alpha=0.5, label='Count')
        # To add values to plot
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                plt.text(i, j, cm[i][j], ha='center', va='center', color='r')
        ax1.set_xlabel('Predicted value')
        ax1.set_ylabel('Observed value')
        # Save
        plt.savefig('./output/thrombolysis_outcome_confusion_matrix.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Get AUC
        auc = roc_auc_score(self.y_test.astype(np.int8), y_probs,
                            multi_class='ovo', average='macro') 
        
        print(f'Outcome multiclass ROC AUC {auc:.3f}')


    
    def train_model(self):

        """
        Train model, get SHAP values, and estimate benchmark thrombolysis rates
        """

        # Define and Fit model
        self.model = XGBClassifier(verbosity=0, seed=42)
        self.model.fit(self.X_train_one_hot, self.y_train)

        # Pickle model and save to pickled_models folder
        pickle.dump(self.model, open('./pickled_models/thrombolysis_choice_model.pkl', 'wb'))
    