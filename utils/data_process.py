import numpy as np
import pandas as pd


class DataProcess:

    """
    Loads data ready for models.

    Attributes:

    full_data:
        Pandas dataframe of full SSNAP data (Cleaned)


    Methods:

    """

    def __init__(self, year_min=2016, year_max=2021):
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
            'scan_to_thrombolysis_time',
            'discharge_disability'
        ]

        self.year_min = year_min
        self.year_max = year_max

        # Load full data
        self.full_data = pd.read_csv('./data/data_for_models.csv', low_memory=False)

        # Restrict years
        mask = ((self.full_data['year'] >= self.year_min) & (self.full_data['year'] <= self.year_max))
        self.full_data = self.full_data[mask]

        # Limit to ambulance arrivals if required
        if self.limit_to_ambo:
            mask = self.full_data['arrive_by_ambulance'] == 1
            self.full_data = self.full_data[mask]



    def create_ml_data(self):
        """
        Added data processing for thrombolysis choice and outcome models
        """
        # Copy full data
        self.ml_data = self.full_data.copy()

        # Limit to 4 hours onset to arrival
        mask = self.ml_data['onset_to_arrival_time'] <= 240
        self.ml_data = self.ml_data[mask]

        # Limit to 6 hours arrival to scan
        mask = self.ml_data['arrival_to_scan_time'] <= 360
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
        print(f'All rows: {len_all}, ML rows:{len_ml}, Fraction: {frac:0.2f}')

    
    def calculate_paramters_for_pathway_simulation(self):
        """
        Calculate parameters for pathway simulation
        """

            # Set up results lists
        stroke_team = []
        admissions = []
        age_80_plus = []
        onset_known = []
        known_arrival_within_4hrs = []
        onset_arrival_mins_mu = []
        onset_arrival_mins_sigma = []
        scan_within_4_hrs = []
        arrival_scan_arrival_mins_mu = []
        arrival_scan_arrival_mins_sigma = []
        onset_scan_4_hrs = []
        scan_needle_mins_mu = []
        scan_needle_mins_sigma = []
        thrombolysis_rate = []
        eligible = []

        # Split data by stroke team
        groups = self.full_data.groupby('stroke_team')
        group_count = 0
        for index, group_df in groups: # each group has an index + dataframe of data
            group_count += 1

            # Record stroke team
            stroke_team.append(index)

            # Record yearly admission numbers
            admissions.append(group_df.shape[0] / (self.year_max - self.year_min + 1))

            # Get thrombolysis rate
            thrombolysis_rate.append(group_df['thrombolysis'].mean())

            # Record onset known proportion and remove rest
            onset_known.append(group_df['onset_known'].mean())
            group_df = group_df[group_df['onset_known'] == 1]

            # Record onset <4 hours and remove rest
            mask = group_df['onset_to_arrival_time'] <= 240
            known_arrival_within_4hrs.append(mask.mean())
            group_df = group_df[mask]
            
            # Calc proportion 80+ (of those arriving within 4 hours)
            over_80 = group_df['age'] >= 80
            age_80_plus.append(over_80.mean())

            # Log mean/sd of onset to arrival
            ln_onset_to_arrival = np.log(group_df['onset_to_arrival_time'])
            onset_arrival_mins_mu.append(ln_onset_to_arrival.mean())
            onset_arrival_mins_sigma.append(ln_onset_to_arrival.std())

            # Record scan within 4 hours of arrival (and remove the rest)
            mask = group_df['arrival_to_scan_time'] <= 240
            scan_within_4_hrs.append(mask.mean())
            group_df = group_df[mask]

            # Log mean/sd of arrival to scan
            ln_arrival_to_scan = np.log(group_df['arrival_to_scan_time'])
            arrival_scan_arrival_mins_mu.append(ln_arrival_to_scan.mean())
            arrival_scan_arrival_mins_sigma.append(ln_arrival_to_scan.std())

            # Record onset to scan in 4 hours and remove rest
            mask = (group_df['onset_to_arrival_time'] + 
                    group_df['arrival_to_scan_time']) <= 240
            onset_scan_4_hrs.append(mask.mean())
            group_df = group_df[mask]

            # Thrombolysis given (to remaining patients)
            eligible.append(group_df['thrombolysis'].mean())

            # Scan to needle
            mask = group_df['scan_to_thrombolysis_time'] > 0
            scan_to_needle = group_df['scan_to_thrombolysis_time'][mask]
            ln_scan_to_needle = np.log(scan_to_needle)
            scan_needle_mins_mu.append(ln_scan_to_needle.mean())
            scan_needle_mins_sigma.append(ln_scan_to_needle.std())

        # Store in DataFrame
        df = pd.DataFrame()
        df['stroke_team'] = stroke_team
        df['thrombolysis_rate'] = thrombolysis_rate
        df['admissions'] = admissions
        df['80_plus'] = age_80_plus
        df['onset_known'] = onset_known
        df['known_arrival_within_4hrs'] = known_arrival_within_4hrs
        df['onset_arrival_mins_mu'] = onset_arrival_mins_mu
        df['onset_arrival_mins_sigma'] = onset_arrival_mins_sigma
        df['scan_within_4_hrs'] = scan_within_4_hrs
        df['arrival_scan_arrival_mins_mu'] = arrival_scan_arrival_mins_mu
        df['arrival_scan_arrival_mins_sigma'] = arrival_scan_arrival_mins_sigma
        df['onset_scan_4_hrs'] = onset_scan_4_hrs
        df['eligable'] = eligible
        df['scan_needle_mins_mu'] = scan_needle_mins_mu
        df['scan_needle_mins_sigma'] = scan_needle_mins_sigma

        # Save to csv
        self.pathway_simulation_parameters = df
        self.pathway_simulation_parameters.to_csv(
            './data/data_for_sim/data_for_sim.csv', index=False)
    












