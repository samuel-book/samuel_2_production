import matplotlib.pyplot as plt
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
    XGBoost model that learns the thrombolysis decisions each hopsital makes on each patient
    (limited to arrivals within 4 hours of known stroke onset).

    The 25 hospitals with the highest hospital SHAP (those hospitals with the highest propensity to
    use thrombolysis) as classed as 'benchmark' hospitals. Patients from from all hopsitals have
    thrombolysis decisions predicted for each of these benchmark hopsitals. A majority vote of those
    benchmark hospitals is taken as a 'benchmark decision' for that patient. For each hopsital the
    proportion of their own patients who have a positive benchmark decision is recorded.


    Model reference:
    ----------------

    For more info see:

    Pearn K, Allen M, Laws A, Monks T, Everson R, James M. (2023) What would other emergency stroke
    teams do? Using explainable machine learning to understand variation in thrombolysis practice.
    European Stroke Journal. https://doi.org/10.1177/23969873231189040

    GitHub Pages on model background: https://samuel-book.github.io/samuel_shap_paper_1/


    Model info
    ----------

    The XGBoost model is based on the following patient features:

    * stroke_team
    * onset_to_arrival_time
    * onset_during_sleep
    * arrival_to_scan_time
    * infarction
    * stroke_severity
    * precise_onset_known
    * prior_disability
    * afib_anticoagulant
    * age

    The model is trained on 75% of the data, and predictions are made for 25% test set (apart from
    benchmark decisions, which are made for all patients).


    Model outputs
    -------------

    The following csv files as saved to the output folder.

     * benchmark_thrombolysis_rates: Records the observed and predicted benchmark thrombolysis rates
     for each hopsital. Results are based on all data (combined trainign and test sets).

     * thrombolysis_choice_feature_values: Input values for the test set used to predict
     thrombolysis decisions.

    * thrombolysis_choice_hospital_shap: Average osiptal SHAP for each hospital.

    * thrombolysis_choice_shap: All SHAP values for the test set.

    * thrombolysis_choice_test_predictions: Model predictions (probability and classification) and
    observed thrombolysis for test set.


    Methods:
    --------

    * __init__: Load data for modelling, get observed thrombolysis rates, split into X and y, and
    one-hot encode stroke team.

    * estimate_benchmark_rates: Estimate thrombolysis rate for each if decision made by a majority
    vote of benchmark hospitals (those with highest hospital SHAP). Uses all patients.

    * get_shap: Fits a SHAP tree explainer to model (without background data), get SHAP values,
    record total SHAP (and convert to probability), get average hospital SHAP per hospital.

    * run: calls training of model, gettign SHAP values, and estimating of benchmark thrombolysis
    rates.

    * train_model: fit XGBoost model, and measure accuracy (accuracy, balanced accuracy, ROC-AUC,
    and compare predicted to observed thrombolysis rates).


    Attributes
    ----------

    Data:

    * stroke_teams: List of all stroke teams (alphabetically sorted)

    * X: all X data (see Model info above)

    * y: all y data: use of thrombolysis (0/1)

    * X_train, X_test, y_train, y_test: Train/test splits based on 75/25 split stratified by
    stroke_team and use of thrombolysis

    * X_train_one_hot, X_test_one_hot: Data with one-hot encoding of stroke team

    Model:

    * model: XGBoost classifier

    * explainer: SHAP explainer model

    * shap_values_extended: Full SHAP (including base value and feature values)

    * shap_values: SHAP values from shap_values_extended

    Outputs:

    * y_pred_proba_df: observed class, predicted class, and predicted
    probabilities for test set

    * hospital_mean_shap: Average hospital SHAP for each hospital

    * shap_values_df: Feature values, SHAP base value, and all SHAP values

    * benchmark_thrombolysis: observed and predicted benchmark thrombolysis use for each hospital
    """

    def __init__(self):
        """
        Load data for modelling, get observed thrombolysis rates, split into X and y, and one-hot
        encode stroke team.
        """

        # Load data
        data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)

        self.thrombolysis_rates = data.groupby('stroke_team').mean()['thrombolysis']
        self.thrombolysis_rates.sort_index(inplace=True)

        # Get X and y
        self.X_fields = [
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
        ]

        self.stroke_teams = list(data['stroke_team'].unique())
        self.stroke_teams.sort()

        self.X = data[self.X_fields]
        self.y = data['thrombolysis']

        # Split 75:25
        strat = data['stroke_team'].map(str) + '-' + data['thrombolysis'].map(str)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.25, stratify=strat, random_state=42)

        # One hot encode hospitals
        X_train_hosp = pd.get_dummies(self.X_train['stroke_team'], prefix='team')
        self.X_train_one_hot = pd.concat([self.X_train, X_train_hosp], axis=1)
        self.X_train_one_hot.drop('stroke_team', axis=1, inplace=True)
        X_test_hosp = pd.get_dummies(self.X_test['stroke_team'], prefix='team')
        self.X_test_one_hot = pd.concat([self.X_test, X_test_hosp], axis=1)
        self.X_test_one_hot.drop('stroke_team', axis=1, inplace=True)
        self.X_test.to_csv('./output/thrombolysis_choice_feature_values.csv')


    def estimate_benchmark_rates(self):
        """
        Estimate thrombolysis rate for each if decision made by a majority vote of benchmark
        hospitals (those with highest hospital SHAP). Uses all patients.
        """

        mask = self.hospital_mean_shap['benchmark'] == 1
        benchmark_hospitals = list(self.hospital_mean_shap[mask].index)
        all_X = pd.concat([self.X_train_one_hot, self.X_test_one_hot])

        results = dict()

        # Loop through each hospital and get their patients
        for hospital in self.stroke_teams:
            mask = all_X[f'team_{hospital}'] == 1
            selected_data = all_X[mask].copy()
            # Remove hospital one hot encode
            selected_data[f'team_{hospital}'] = False
            # Loop through benchmark hospitals
            decisions = []
            for benchmark_hosp in benchmark_hospitals:
                # Change one-hot encoding
                selected_data[f'team_{benchmark_hosp}'] = True
                # Get predictions
                decisions.append(self.model.predict(selected_data))
                # Reset hospital
                selected_data[f'team_{benchmark_hosp}'] = False
            # Get majority vote
            decisions = np.array(decisions)
            benchmark = decisions.mean(axis=0) >= 0.5
            results[hospital] = np.mean(benchmark)

        self.benchmark_thrombolysis = \
            pd.DataFrame.from_dict(results, orient='index', columns=['benchmark'])
        self.benchmark_thrombolysis.sort_index(inplace=True)
        self.benchmark_thrombolysis['observed'] = self.thrombolysis_rates
        self.benchmark_thrombolysis = self.benchmark_thrombolysis.round(3)
        self.benchmark_thrombolysis.to_csv('./output/benchmark_thrombolysis_rates.csv')

    def get_shap(self):

        """
        Fits a SHAP tree explainer to model (without background data), get SHA values, record total
        SHAP (and convert to probability), get average hospital SHAP per hospital.
        """

        # Get SHAP valuess
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values_extended = self.explainer(self.X_test_one_hot)
        self.shap_values = self.shap_values_extended.values
        self.shap_values_df = pd.DataFrame(
            self.shap_values, columns=list(self.X_test_one_hot))

        # Sum hospital SHAPs
        teams = [hosp for hosp in list(self.X_train_one_hot) if hosp[0:4] == 'team']
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
        odds = np.exp(self.shap_values_df['total'])
        self.shap_values_df['probability'] = odds / (1 + odds)

        # Get average hospital SHAP values
        self.shap_values_df['stroke_team'] = self.X_test['stroke_team'].values
        self.hospital_mean_shap = pd.DataFrame()
        self.hospital_mean_shap['hospital_SHAP'] = \
            self.shap_values_df.groupby('stroke_team').mean()['hospital']
        # Identify and label top 25 benchmark hospitals
        self.hospital_mean_shap.sort_values(
            by='hospital_SHAP', ascending=False, inplace=True)
        benchmark = np.zeros(len(self.hospital_mean_shap))
        benchmark[0:25] = 1
        self.hospital_mean_shap['benchmark'] = benchmark
        self.hospital_mean_shap.to_csv(
            './output/thrombolysis_choice_hospital_shap.csv')

        # Save
        self.shap_values_df.drop('stroke_team', axis=1, inplace=True)
        self.shap_values_df = self.shap_values_df.round(4)
        self.shap_values_df.to_csv('./output/thrombolysis_choice_shap.csv')

    def plot_hospital_shap(self):

        hospital_shap = self.hospital_mean_shap['hospital_SHAP'].values
        max_scale = max(max(hospital_shap), -min(hospital_shap))

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        ax.hist(hospital_shap, bins=np.arange(-max_scale, max_scale+0.01, 0.1))
        ax.set_xlabel('Hospital SHAP value')
        ax.set_ylabel('Count')
        plt.savefig('./output/thrombolysis_choice_hopsital_shap.jpg', dpi=300,
                    bbox_inches='tight', pad_inches=0.2)
        
        plt.close()


    def plot_shap_scatter(self):

        feat_to_show = self.X_fields.copy()
        feat_to_show.remove('stroke_team')

        fig = plt.figure(figsize=(12, 12))
        for n, feat in enumerate(feat_to_show):
            ax = fig.add_subplot(3, 3, n+1)
            shap.plots.scatter(self.shap_values_extended[:, feat], x_jitter=0, ax=ax, show=False)

            # Add line at Shap = 0
            ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], c='0.5')

            ax.set_ylabel(f'SHAP value (log odds) for\n{feat}')
            ax.set_title(feat)

        plt.tight_layout(pad=2)

        fig.savefig('output/thrombolysis_choice_shap_scatter.jpg',
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        
        plt.close()


    def run(self):
        """
        Train model, get SHAP values, and estimate benchmark thrombolysis rates
        """

        self.train_model()
        self.get_shap()
        self.estimate_benchmark_rates()
        self.plot_shap_scatter()
        self.plot_hospital_shap()

    def train_model(self):
        """
        Fit XGBoost model, and measure accuracy (accuracy, balanced accuracy, ROC-AUC, and compare
        predicted to observed thrombolysis rates).
        """

        # Define and Fit model
        self.model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        self.model.fit(self.X_train_one_hot, self.y_train)

        # Pickle model and save to pickled_models folder
        pickle.dump(self.model, open('./pickled_models/thrombolysis_choice_model.pkl', 'wb'))
        
        # Get predictions for test set
        y_pred_proba = self.model.predict_proba(self.X_test_one_hot)[:, 1]
        y_pred = y_pred_proba >= 0.5
        self.y_pred_proba_df = pd.DataFrame()
        self.y_pred_proba_df['probability'] = y_pred_proba
        self.y_pred_proba_df['predicted'] = y_pred * 1
        self.y_pred_proba_df['observed'] = self.y_test.values

        self.y_pred_proba_df.to_csv('./output/thrombolysis_choice_test_predictions.csv')
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
