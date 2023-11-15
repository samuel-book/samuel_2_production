import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataProcess:
    
    """
    Loads data ready for models.

    Attributes:

    full_data:
        Pandas dataframe of full SSNAP data (Cleaned)
        

    Methods:
        
    """
    
    def __init__(self):
        """
        Creates the data load object
        """

        # Default to limit to ambulance arrivals
        self.limit_to_ambo = True

        # List fields for ML; some are created in `create_ml_data`
        self.machine_learning_fields = [
            'stroke_team',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'arrival_to_scan_time',
            'onset_to_scan',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'any_afib_diagnosis',
            'afib_anticoagulant',
            'age',
            'onset_to_thrombolysis',
            'thrombolysis',
            'discharge_disability'
        ]

        self.year_min = 2016
        self.year_max = 2025
        
        # Load stored data
        self.load_data()

    
    def load_data(self):
        """
        Load data and create datasets
        """

        # Load full data
        self.full_data = pd.read_csv(
            './data/data_for_models.csv', low_memory=False)

        # Restrict years
        mask = ((self.full_data['year'] >= self.year_min) &
                (self.full_data['year'] <= self.year_max))
        self.full_data = self.full_data[mask]
        

    def create_ml_data(self):
        """
        Added data processing for thrombolysis choice and outcome models
        """

        # Limit to ambulance arrivals if required
        if self.limit_to_ambo:
            mask = self.full_data['arrive_by_ambulance'] == 1
            self.ml_data = self.full_data[mask]
        else:
            self.ml_data = self.full_data.copy()

        # Limit to 4 hours onset to arrival
        mask = self.ml_data['onset_to_arrival_time'] <= 240
        self.ml_data = self.ml_data[mask]

        # Limit to known discharge disability
        mask = self.ml_data['discharge_disability'] >= 0
        self.ml_data = self.ml_data[mask]

        # Replace afib_anticoagulant NaN with 0
        self.ml_data['afib_anticoagulant'].fillna(0, inplace=True)

        # Limit to patients who have had scan
        mask = self.ml_data['arrival_to_scan_time'] >= 0
        self.ml_data = self.ml_data[mask]

        # Calculate onset to scan
        def f(row):
            return row['arrival_to_scan_time'] + row['onset_to_arrival_time']
        self.ml_data['onset_to_scan'] = self.ml_data.apply(f, axis=1)

        # Calculate onset to thrombolysis (return -10 for no thrombolysis)
        def f(row):
            if row['scan_to_thrombolysis_time'] >= 0:
                return row['scan_to_thrombolysis_time'] + row['onset_to_arrival_time']
            else:
                return -10
        self.ml_data['onset_to_thrombolysis'] = self.ml_data.apply(f, axis=1)

        # Get any diagnosis of atrial fibrillation
        def f(row):
            if (row['atrial_fibrillation'] == 1) or (row['new_afib_diagnosis'] == 1):
                return 1
            else:
                return 0
        self.ml_data['any_afib_diagnosis'] = self.ml_data.apply(f, axis=1)

        # Limit to required fields
        self.ml_data = self.ml_data[self.machine_learning_fields]

        # Shuffle data
        self.ml_data = self.ml_data.sample(
            frac=1, random_state=42).reset_index(drop=True)

        # Save complete data
        self.ml_data.to_csv('./data/data_for_ml/complete_ml_data.csv', index=False)

        # Print lengths of output
        len_all = len(self.full_data)
        len_ml = len(self.ml_data)
        frac = len_ml / len_all
        print (f'All rows: {len_all}, ML rows:{len_ml}, Fraction: {frac:0.2f}')