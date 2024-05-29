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

    def __init__(self, remove_afib_anticoagulant=False):
        """
        """

        # Load data
        self.data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)
               
        # Remove infarction = 0
        self.data = self.data[self.data['infarction'] == 1]

        # Remove afib_anticoagulant =1
        if remove_afib_anticoagulant:
            self.data = self.data[self.data['afib_anticoagulant'] == 0]

        # Limit to patients with onset to scan of no more than 4 hr 15 minutes
        self.data = self.data[self.data['onset_to_scan'] <= 255]
        
        self.hospital_stats = pd.read_csv(
            './output/hospital_stats_4hr_arrivals.csv', index_col='stroke_team')

        # Add simulated onset to thrombolysis time for all patients
        hospital_scan_to_thrombolysis = self.hospital_stats['scan_to_thrombolysis_time'].to_dict()
        self.data['hospital_scan_to_thrombolysis_time'] = self.data['stroke_team'].map(hospital_scan_to_thrombolysis)

        self.data['simulated_onset_to_thrombolysis'] = (
            self.data['onset_to_scan'] +
            self.data['hospital_scan_to_thrombolysis_time'])

        # Get X and y
        self.X_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis']

        self.stroke_teams = list(self.data['stroke_team'].unique())
        self.stroke_teams.sort()

        self.X = self.data[self.X_fields]
        self.y = self.data['discharge_disability'].values

        # Split 80:20
        strat = self.data['discharge_disability'].values
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.20, stratify=strat, random_state=42)

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


    def predict_all_patients(self):
        """
        Make predictions with and without thrombolysis for all patients and compare outcomes.
        When thrombolysis not used in actual patient use apply hopsital median scan-to-thrombolysis time
        """

        # Test with all onset_to_thrombolysis set to -10 (no thrombolysis)
        results = pd.concat([self.X_train, self.X_test])
        data_copy = pd.concat([self.X_train_one_hot, self.X_test_one_hot])
        results['thrombolysis_given'] = 1.0 * data_copy['onset_to_thrombolysis'] >= 0
        data_copy['onset_to_thrombolysis'] = -10
        self.all_patients_outcomes_untreated = self.model.predict_proba(data_copy)
        self.all_patients_outcomes_untreated_weighted_mrs = \
            (self.all_patients_outcomes_untreated * np.arange(7)).sum(axis=1)
        self.all_patients_outcomes_untreated_0_to_4 = self.all_patients_outcomes_untreated[:,0:5].sum(axis=1)
        results['untreated_weighted_mrs'] = self.all_patients_outcomes_untreated_weighted_mrs
        results['untreated_0_to_4'] = self.all_patients_outcomes_untreated_0_to_4

        # Test with all onset_to_thrombolysis set to simulated onset_to_thrombolysis
        data_copy = pd.concat([self.X_train_one_hot, self.X_test_one_hot])
        data_copy['onset_to_thrombolysis'] = self.data['simulated_onset_to_thrombolysis']
        self.all_patients_outcomes_treated = self.model.predict_proba(data_copy)
        self.all_patients_outcomes_treated_weighted_mrs = \
            (self.all_patients_outcomes_treated * np.arange(7)).sum(axis=1)
        self.all_patients_outcomes_treated_0_to_4 = self.all_patients_outcomes_treated[:,0:5].sum(axis=1)
        results['treated_weighted_mrs'] = self.all_patients_outcomes_treated_weighted_mrs
        results['treated_0_to_4'] = self.all_patients_outcomes_treated_0_to_4
    
        # Check for improved outcome
        self.all_patients_outcomes_improved = (
            (self.all_patients_outcomes_treated_weighted_mrs <= self.all_patients_outcomes_untreated_weighted_mrs) &
            (self.all_patients_outcomes_treated_0_to_4 >= self.all_patients_outcomes_untreated_0_to_4))
        results['improved_outcome'] = self.all_patients_outcomes_improved

        # Compare outcome with thrombolysis given
        results['thrombolysis_given_agrees_with_improved_outcome'] = (
            results['thrombolysis_given'] == results['improved_outcome'])
        results['TP'] = (
            (results['thrombolysis_given'] == 1) & (results['improved_outcome'] == 1))
        results['FP'] = (
            (results['thrombolysis_given'] == 1) & (results['improved_outcome'] == 0))
        results['FN'] = (
            (results['thrombolysis_given'] == 0) & (results['improved_outcome'] == 1))
        results['TN'] = (
            (results['thrombolysis_given'] == 0) & (results['improved_outcome'] == 0))

        # Store results
        results.to_csv('./output/thrombolysis_outcome_predictions.csv', index=False)
        self.all_patients_outcomes = results

        # Summarise results by stroke_team
        results_by_team = results.groupby('stroke_team').mean()
        results_by_team['sensitivity'] = \
            results_by_team['TP'] / (results_by_team['TP'] + results_by_team['FN'])
        results_by_team['specificity'] = \
            results_by_team['TN'] / (results_by_team['TN'] + results_by_team['FP'])
        
        # Add stroke team ranks for sensitivity and specificity
        results_by_team['sensitivity_rank'] = results_by_team['sensitivity'].rank(ascending=False)
        results_by_team['specificity_rank'] = results_by_team['specificity'].rank(ascending=False)
        
        # Store results by team
        results_by_team.to_csv('./output/thrombolysis_outcome_predictions_by_team.csv')



     
    def run(self):
        """
        Train model, get SHAP values, and estimate benchmark thrombolysis rates
        """

        self.train_model()
        self.test_model()
        self.predict_all_patients()

    
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
        ax1.set_xlabel('Predicted mRS')
        ax1.set_ylabel('Observed mRS')
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
        pickle.dump(self.model, open('./pickled_models/thrombolysis_outcome_model.pkl', 'wb'))
    