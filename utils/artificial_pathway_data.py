import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class ArtificialPathwayData:
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
                './data/data_for_models.csv', low_memory=False)
        
        # Load rename_dict 
        import pickle
        with open('./data/artificial_ml_data/rename_dict.pkl', 'rb') as f:
            self.rename_dict = pickle.load(f)

        # Limit data to stroke team in rename dict
        self.full_data = self.full_data[self.full_data['stroke_team'].isin(self.rename_dict.keys())]

        # Rename stroke teams
        self.full_data['stroke_team'] = self.full_data['stroke_team'].map(self.rename_dict)

        # Get stroke teams
        self.stroke_teams = self.full_data['stroke_team'].unique()

    
    def create_artificial_pathway_data(self, patients_per_hospital=500):
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

            # Add arrive by ambulance by sampling from data
            sample = np.random.choice(data['arrive_by_ambulance'], patients_per_hospital)
            df['arrive_by_ambulance'] = sample.astype(int)

            # Sample age
            sample = np.random.choice(data['age'], patients_per_hospital)
            df['age'] = sample

            # onset_to_arrival_time
            sample = np.random.choice(data['onset_to_arrival_time'], patients_per_hospital)
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['onset_to_arrival_time'] = sample + 5

            # If onset to arrival time is greater than 0 then set onset_known to 1 otherwise 0
            mask = df['onset_to_arrival_time'] > 0
            df['onset_known'] = 0
            df.loc[mask, 'onset_known'] = 1

            # Sample arrival_to_scan_time
            sample = np.random.choice(data['arrival_to_scan_time'], patients_per_hospital)
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['arrival_to_scan_time'] = sample + 5

            # Sample infarction
            sample = np.random.choice(data['infarction'], patients_per_hospital)
            df['infarction'] = sample

            # Get thrombolysis rate for treatable pop
            mask = ((data['onset_to_arrival_time'] <= 240) & 
                    (data['onset_to_arrival_time'] + data['arrival_to_scan_time'] <= 255) &
                    (data['infarction'] == 1))
            thrombolysis_rate = data[mask]['thrombolysis'].mean()
            # Sample thrombolysis
            mask = ((df['onset_to_arrival_time'] <= 240) & 
                    (df['onset_to_arrival_time'] + df['arrival_to_scan_time'] <= 255) &
                    (df['infarction'] == 1))
            sample = np.random.binomial(1, thrombolysis_rate, mask.sum())
            df['thrombolysis'] = 0
            df['thrombolysis'].loc[mask] = sample

            # Sample scan_to_thrombolysis_time when thrombolysis is 1
            mask = data['thrombolysis'] == 1
            scan_to_thrombolysis = data[mask]['scan_to_thrombolysis_time']
            mask = df['thrombolysis'] == 1
            sample = np.random.choice(scan_to_thrombolysis, mask.sum())
            # Round to the closest 5
            sample = np.round(sample / 5) * 5
            df['scan_to_thrombolysis_time'] = np.nan
            df['scan_to_thrombolysis_time'].loc[mask] = sample


            generated_data.append(df)


        # Concatenate data, reindex, and shuffle
        generated_data = pd.concat(generated_data)
        generated_data = generated_data.sample(frac=1)
        generated_data = generated_data.reset_index(drop=True)

        # Save
        generated_data.to_csv('./data/artificial_ml_data/artificial_patient_pathway_data.csv', index=False)

    





