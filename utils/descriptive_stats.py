import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DescriptiveStatistics:
    """
    Generate descriptive statistics, and save to output
    """

    def __init__(self, year_min=2016, year_max=2021):
        """
        Load data and restrict years.
        """

        self.year_min = year_min
        self.year_max = year_max

        # Load full data
        self.full_data = pd.read_csv('./data/data_for_models.csv', low_memory=False)
        
        # Restrict years
        mask = ((self.full_data['year'] >= self.year_min) &
                (self.full_data['year'] <= self.year_max))
        self.full_data = self.full_data[mask]

        # Add fields
        self.full_data['onset_within_4hrs'] = \
            self.full_data['onset_to_arrival_time'] <= 240
        
        # Save completeion
        complete = pd.DataFrame()
        complete['complete'] = self.full_data.count()/len(self.full_data)
        complete = complete.round(3)
        complete.to_csv('./output/full_data_complete.csv', index_label='field')

        # Get stroke teams
        self.stroke_teams = list(self.full_data['stroke_team'].unique())
        self.stroke_teams.sort()

        # List fields for averages
        self.fields_for_average_statistics = {
            'age': 'mean',
            'male': 'mean',
            'onset_known': 'mean',
            'onset_within_4hrs': 'mean',
            'precise_onset_known': 'mean',
            'onset_during_sleep': 'mean',
            'arrive_by_ambulance': 'mean',
            'prior_disability': 'mean',
            'stroke_severity': 'mean',
            'death': 'mean',
            'discharge_disability': 'mean',
            'thrombolysis': 'mean',
            'call_to_ambulance_arrival_time': 'median',
            'ambulance_on_scene_time': 'median',
            'ambulance_travel_to_hospital_time': 'median',
            'arrival_to_scan_time': 'median',
            'scan_to_thrombolysis_time': 'median',
            'onset_to_arrival_time': 'median'
        }

        # Summary
        self.stats_summary = dict()
        self.stats_summary['min year'] = self.full_data['year'].min()
        self.stats_summary['max year'] = self.full_data['year'].max()
        self.stats_summary['total records'] = len(self.full_data)
        self.stats_summary['4 hr arrivals'] = self.full_data['onset_within_4hrs'].mean()
        self.stats_summary = pd.DataFrame.from_dict(self.stats_summary, orient='index')
        self.stats_summary.index.name='field'
        self.stats_summary.to_csv('./output/stats_summary.csv')


    def calculate_average_statistics(self):

        self.hopsital_stats_all_arrivals = dict()

        grouped = self.full_data.groupby('stroke_team')
        for name, group in grouped:
            stats = dict()
            stats['admissions'] = len(group)
            for field in self.fields_for_average_statistics.keys():
                if self.fields_for_average_statistics[field] == 'mean':
                    stats[field] = group[field].mean()
                elif self.fields_for_average_statistics[field] == 'median':
                    stats[field] = group[field].median()
            self.hopsital_stats_all_arrivals[name] = stats

        # Convert to DataFrame
        self.hopsital_stats_all_arrivals = \
                pd.DataFrame.from_dict(self.hopsital_stats_all_arrivals).T.round(2)
        
        
    def run(self):
        self.calculate_average_statistics()