import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class ArtificialPatientData:
    """
    Creates and tests artificial patient data. 
        
    The data is created from sampling os patient data fields independently.
   """
    
    def __init__(self, patients_per_hopsital=500):
        """
        Constructor for artificial patient data.
        """

        # Load full data
        self.full_data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)

        # Get list of stroke teams and sort
        self.stroke_teams = self.full_data['stroke_team'].unique()
        self.stroke_teams.sort()

        # X fields for models
        self.X_thrombolysis_fields = [
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

        self.X_outcome_fields = [
            'prior_disability',
            'stroke_severity',
            'stroke_team',
            'onset_to_thrombolysis',
            'age',
            'precise_onset_known',
            'any_afib_diagnosis',
            'infarction']

        # Fit models
        self.train_thrombolysis_choice_model()
        self.train_outcome_model()

    
    def create_artificial_data(self, patients_per_hospital=500):
        """
        Creates artificial patient data.
        """

        generated_data = []

        # Create data for each stroke team
        for stroke_team in self.stroke_teams:

            df = pd.DataFrame()

            # Get data for this stroke team
            data = self.full_data[self.full_data['stroke_team'] == stroke_team]

            # Add stroke team
            df['stroke_team'] = [stroke_team] * patients_per_hospital

            # Sample age
            sample = np.random.choice(data['age'], patients_per_hospital)
            df['age'] = sample

            # onset_to_arrival_time
            sample = np.random.choice(data['onset_to_arrival_time'], patients_per_hospital)
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['onset_to_arrival_time'] = sample


            # onset_during_sleep
            sample = np.random.choice(data['onset_during_sleep'], patients_per_hospital)
            df['onset_during_sleep'] = sample

            # precise_onset_known
            sample = np.random.choice(data['precise_onset_known'], patients_per_hospital)
            df['precise_onset_known'] = sample

            # If onset_during_sleep set to 1, set precise_onset_known to 0
            mask = df['onset_during_sleep'] == 1
            df.loc[mask, 'precise_onset_known'] = 0

            # arrival_to_scan_time
            sample = np.random.choice(data['arrival_to_scan_time'], patients_per_hospital)
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['arrival_to_scan_time'] = sample

            # infarction
            sample = np.random.choice(data['infarction'], patients_per_hospital)
            df['infarction'] = sample

            # Stroke severity - sample separately for infarction = 0 and 1
            # infarction = 0
            mask = data['infarction'] == 0
            infarction_0_sample = np.random.choice(
                 data[mask]['stroke_severity'], patients_per_hospital)
            mask = data['infarction'] == 1
            infarction_1_sample = np.random.choice(
                 data[mask]['stroke_severity'], patients_per_hospital)
            # Choose which sample to use based on df['infarction'] value
            sample = np.where(df['infarction'] == 0, infarction_0_sample, infarction_1_sample)
            df['stroke_severity'] = sample

            # prior_disability
            sample = np.random.choice(data['prior_disability'], patients_per_hospital)
            df['prior_disability'] = sample

            # afib_anticoagulant
            sample = np.random.choice(data['afib_anticoagulant'], patients_per_hospital)
            df['afib_anticoagulant'] = sample
            df['any_afib_diagnosis'] = sample

            # sample scan to thrombolysis from scan_to_thrombolysis_time when >0
            # Later set all non thrombolysed to -10
            mask = data['scan_to_thrombolysis_time'] > 0
            sample = np.random.choice(
                data[mask]['scan_to_thrombolysis_time'], patients_per_hospital)
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['scan_to_thrombolysis_time'] = sample

            # Set onset to thrombolysis time
            df['onset_to_thrombolysis'] = (df['onset_to_arrival_time'] + 
                                           df['arrival_to_scan_time'] +
                                           df['scan_to_thrombolysis_time'])
            
            # Limit onset to thrombolysis time to 285 mins
            mask = df['onset_to_thrombolysis'] > 285
            df.loc[mask, 'onset_to_thrombolysis'] = 285
 
            generated_data.append(df)

        # Concatenate data, reindex, and shuffle
        generated_data = pd.concat(generated_data)
        generated_data = generated_data.sample(frac=1)
        generated_data = generated_data.reset_index(drop=True)

        # Apply thrombolysis decision model
        X = generated_data[self.X_thrombolysis_fields]
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)
        y_probs = self.choice_model.predict_proba(X_one_hot)[:, 1]
        # Choose thrombolysis used based on probability
        generated_data['thrombolysis'] = np.random.binomial(1, y_probs)
        # Ensure all non-ischaemic stroke do not receive thrombolysis
        mask = generated_data['infarction'] == 0
        generated_data.loc[mask, 'thrombolysis'] = 0

        # When thrombolysis is 0 set onset_to_thrombolysis_time to -10, 
        mask = generated_data['thrombolysis'] == 0
        generated_data.loc[mask, 'onset_to_thrombolysis'] = -10
        # When thrombolysis is 0 set scan_to_thrombolysis_time to empty
        generated_data.loc[mask, 'scan_to_thrombolysis_time'] = np.nan        

        # Apply outcome model
        X = generated_data[self.X_outcome_fields]
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)
        y_probs = self.outcome_model.predict_proba(X_one_hot)
        # Choose discharge disability based on array probabilities
        outcomes = np.arange(0, 7)
        sampled_outcome = []
        for i in range(len(generated_data)):
            sampled_outcome.append(np.random.choice(outcomes, p=y_probs[i]))
        generated_data['discharge_disability'] = sampled_outcome

        # Shuffle
        generated_data = generated_data.sample(frac=1)

        self.generated_data = generated_data

        # Save
        generated_data.to_csv('./data/artificial_ml_data/artificial_ml_data.csv', index=False)

    
    def train_outcome_model(self):
            
            # Limit to thrombectomy = 0
            mask = self.full_data['thrombectomy'] == 0
            
            # Get X and y
            X = self.full_data[mask][self.X_outcome_fields]
            y = self.full_data[mask]['discharge_disability']
    
            # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
            encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
            encoder.fit(X[['stroke_team']])
            one_hot_encoded = encoder.transform(X[['stroke_team']])
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=X.index)
            X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
            X_one_hot.drop('stroke_team', axis=1, inplace=True)
    
            # Define and Fit model
            self.outcome_model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
            self.outcome_model.fit(X_one_hot, y)

    
    def train_thrombolysis_choice_model(self):        

        # Get X and y
        X = self.full_data[self.X_thrombolysis_fields]
        y = self.full_data['thrombolysis']

        # One hot encode stroke teams using OneHotEncoder with self.stroke_teams as categories
        encoder = OneHotEncoder(categories=[self.stroke_teams], sparse=False)
        encoder.fit(X[['stroke_team']])
        one_hot_encoded = encoder.transform(X[['stroke_team']])
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=self.stroke_teams, index=X.index)
        X_one_hot = pd.concat([X, one_hot_encoded_df], axis=1)
        X_one_hot.drop('stroke_team', axis=1, inplace=True)

        # Define and Fit model
        self.choice_model = XGBClassifier(verbosity=0, seed=42, learning_rate=0.5)
        self.choice_model.fit(X_one_hot, y)



            





