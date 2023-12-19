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

    def __init__(self, train_models=False, replicates=30):
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
        
        # Get benchmark data
        benchmark_data = pd.read_csv(
            './output/thrombolysis_choice_hospital_shap.csv')        
        mask = benchmark_data['benchmark'] == 1
        benchmark_data = benchmark_data[mask]
        self.benchmark_hospitals = benchmark_data['stroke_team'].values

        # Train new models or load existing models
        if train_models:
            self.train_models(replicates)
        
        # Load models
        self.choice_models = pickle.load(open('./pickled_models/replicate_choice_models.pkl', 'rb'))
        self.outcome_models = pickle.load(open('./pickled_models/replicate_outcome_models.pkl', 'rb'))


    def predict_patient(self, patient_data):

        patient = pd.DataFrame(patient_data, index=[0])

        # Get thrombolysis choice prediction
        fields = self.thrombolysis_choice_fields.copy()
        fields.remove('thrombolysis')
        patient_choice = patient[fields]
        enc = OneHotEncoder(categories=[self.stroke_teams])
        one_hot = enc.fit_transform(patient_choice[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        patient_choice = pd.concat([patient_choice, one_hot], axis=1)
        patient_choice.drop('stroke_team', axis=1, inplace=True)
        thrombolysis_predictions = []
        for i in range(len(self.choice_models)):
            model = self.choice_models[i]
            thrombolysis_predictions.append(model.predict_proba(patient_choice)[:,1])
        thrombolysis_predictions = np.array(thrombolysis_predictions)
        self.thrombolysis_prediction = np.mean(thrombolysis_predictions)
        self.thrombolysis_prediction_std = np.std(thrombolysis_predictions)
        # Calculate 95% CI
        self.thrombolysis_prediction_ci = scipy.stats.norm.interval(
            0.95, loc=self.thrombolysis_prediction, scale=self.thrombolysis_prediction_std)


        # Get benchmark thrombolysis predictions
        benchmark_predictions = []
        for benchmark_hosp in self.benchmark_hospitals:
            p = patient_choice.copy()
            # Change one-hot encoding
            p[f'{benchmark_hosp}'] = True
            # Get predictions
            thrombolysis_predictions = []
            for i in range(len(self.choice_models)):
                model = self.choice_models[i]
                thrombolysis_predictions.append(model.predict_proba(p)[:,1])
            # Reset hospital
            p[f'{benchmark_hosp}'] = False
            # Get mean prediction
            thrombolysis_predictions = np.array(thrombolysis_predictions)
            benchmark_prediction = np.mean(thrombolysis_predictions)
            benchmark_predictions.append(benchmark_prediction)
        
        self.thrombolysis_choice_benchmark_mean = np.mean(benchmark_predictions)
        self.thrombolysis_choice_benchmark_std = np.std(benchmark_predictions)
        # Calculate 95% CI
        self.thrombolysis_choice_benchmark_ci = scipy.stats.norm.interval(
            0.95, loc=self.thrombolysis_choice_benchmark_mean, scale=self.thrombolysis_choice_benchmark_std)

        # Get thrombolysis outcome prediction
        untreated_dist = []
        treated_dist = []
        untreated_less_3 = []
        treated_less_3 = []
        untreated_more_4 = []
        treated_more_4 = []
        untreated_weighted_mrs = []
        treated_weighted_mrs = []
        improvement = []
        fields = self.thrombolysis_outcome_fields.copy()
        fields.remove('discharge_disability')
        p = patient[fields]
        enc = OneHotEncoder(categories=[self.stroke_teams])
        one_hot = enc.fit_transform(p[['stroke_team']]).toarray()
        one_hot = pd.DataFrame(one_hot, columns=self.stroke_teams)
        p_treated = pd.concat([p, one_hot], axis=1)
        p_treated.drop('stroke_team', axis=1, inplace=True)
        p_untreated = p_treated.copy()
        p_untreated['onset_to_thrombolysis'] = -10
    
        for i in range(len(self.outcome_models)):        
        # Get untreated and treated distributions
            untreated = self.outcome_models[i].predict_proba(p_untreated)
            treated = self.outcome_models[i].predict_proba(p_treated)
            untreated_dist.append(untreated)
            treated_dist.append(treated)
            # Get weighted average of mRS scores
            weighted_untreated = np.sum(untreated * np.arange(7))
            weighted_treated = np.sum(treated * np.arange(7))
            untreated_weighted_mrs.append(weighted_untreated)
            treated_weighted_mrs.append(weighted_treated)
            improvement.append(0-(weighted_treated - weighted_untreated))
            # Get untreated and treated distributions for mRS <3
            untreated_less_3.append(np.sum(untreated[:3]))
            treated_less_3.append(np.sum(treated[:3]))
            # Get untreated and treated distributions for mRS >4
            untreated_more_4.append(np.sum(untreated[5:]))
            treated_more_4.append(np.sum(treated[5:]))

        # Get mean predictions
        untreated_dist = np.array(untreated_dist)
        treated_dist = np.array(treated_dist)
        self.untreated_dist = np.mean(untreated_dist, axis=0)
        self.treated_dist = np.mean(treated_dist, axis=0)
        self.untreated_dist_std = np.std(untreated_dist, axis=0)
        self.treated_dist_std = np.std(treated_dist, axis=0)
        # Calculate 95% CI
        self.untreated_dist_ci = scipy.stats.norm.interval(
            0.95, loc=self.untreated_dist, scale=self.untreated_dist_std)
        self.treated_dist_ci = scipy.stats.norm.interval(
            0.95, loc=self.treated_dist, scale=self.treated_dist_std)
        # Get mean predictions for mRS <3
        untreated_less_3 = np.array(untreated_less_3)
        treated_less_3 = np.array(treated_less_3)
        self.untreated_less_3 = np.mean(untreated_less_3)
        self.treated_less_3 = np.mean(treated_less_3)
        self.untreated_less_3_std = np.std(untreated_less_3)
        self.treated_less_3_std = np.std(treated_less_3)
        # Calculate 95% CI
        self.untreated_less_3_ci = scipy.stats.norm.interval(
            0.95, loc=self.untreated_less_3, scale=self.untreated_less_3_std)
        self.treated_less_3_ci = scipy.stats.norm.interval(
            0.95, loc=self.treated_less_3, scale=self.treated_less_3_std)
        # Get mean predictions for mRS >4
        untreated_more_4 = np.array(untreated_more_4)
        treated_more_4 = np.array(treated_more_4)
        self.untreated_more_4 = np.mean(untreated_more_4)
        self.treated_more_4 = np.mean(treated_more_4)
        self.untreated_more_4_std = np.std(untreated_more_4)
        self.treated_more_4_std = np.std(treated_more_4)
        # Calculate 95% CI
        self.untreated_more_4_ci = scipy.stats.norm.interval(
            0.95, loc=self.untreated_more_4, scale=self.untreated_more_4_std)
        self.treated_more_4_ci = scipy.stats.norm.interval(
            0.95, loc=self.treated_more_4, scale=self.treated_more_4_std)
        # Get mean predictions for weighted mRS
        untreated_weighted_mrs = np.array(untreated_weighted_mrs)
        treated_weighted_mrs = np.array(treated_weighted_mrs)
        self.untreated_weighted_mrs = np.mean(untreated_weighted_mrs)
        self.treated_weighted_mrs = np.mean(treated_weighted_mrs)
        # Calculate 95% CI
        self.untreated_weighted_mrs_ci = scipy.stats.norm.interval(
            0.95, loc=self.untreated_weighted_mrs, scale=np.std(untreated_weighted_mrs))
        self.treated_weighted_mrs_ci = scipy.stats.norm.interval(
            0.95, loc=self.treated_weighted_mrs, scale=np.std(treated_weighted_mrs))
        # Get mean predictions for improvement
        improvement = np.array(improvement)
        self.improvement = np.mean(improvement)
        self.improvement_ci = scipy.stats.norm.interval(
            0.95, loc=self.improvement, scale=np.std(improvement))
        

    def train_models(self, replicates):
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





