import pandas as pd


class DescriptiveStatistics:
    """
    Generate descriptive statistics, and save to output
    """

    def __init__(self):
        """
        Load data and restrict years.
        """

        # Load full data
        self.full_data = pd.read_csv('./data/data_for_models.csv', low_memory=False)

        # Add fields
        self.full_data['onset_within_4hrs'] = \
            self.full_data['onset_to_arrival_time'] <= 240
        self.full_data['prior_disability_0-2'] = \
            self.full_data['prior_disability'] < 3
        self.full_data['discharge_disability_0-2'] = \
            self.full_data['discharge_disability'] < 3
        self.full_data['discharge_disability_5-6'] = \
            self.full_data['discharge_disability'] > 4

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
            'prior_disability': 'mean',
            'prior_disability_0-2': 'mean',
            'stroke_severity': 'mean',
            'onset_known': 'mean',
            'onset_to_arrival_time': 'median',
            'onset_within_4hrs': 'mean',
            'precise_onset_known': 'mean',
            'onset_during_sleep': 'mean',
            'arrive_by_ambulance': 'mean',
            'call_to_ambulance_arrival_time': 'median',
            'ambulance_on_scene_time': 'median',
            'ambulance_travel_to_hospital_time': 'median',
            'arrival_to_scan_time': 'median',
            'thrombolysis': 'mean',
            'scan_to_thrombolysis_time': 'median',
            'discharge_disability': 'mean',
            'discharge_disability_0-2': 'mean',
            'discharge_disability_5-6': 'mean',
            'death': 'mean'
            
        }

        # Summary
        self.stats_summary = dict()
        self.stats_summary['min year'] = self.full_data['year'].min()
        self.stats_summary['max year'] = self.full_data['year'].max()
        self.stats_summary['total records'] = len(self.full_data)
        self.stats_summary['4 hr arrivals'] = self.full_data['onset_within_4hrs'].mean()
        self.stats_summary = pd.DataFrame.from_dict(self.stats_summary, orient='index')
        self.stats_summary.index.name = 'field'
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
        self.hopsital_stats_all_arrivals.index.name = 'stroke_team'

        # Save
        self.hopsital_stats_all_arrivals.to_csv('./output/hospital_stats.csv')

        # Repeat for four hour arrivals
        mask = self.full_data['onset_to_arrival_time'] <= 240

        self.hopsital_stats_4hr_arrivals = dict()

        grouped = self.full_data[mask].groupby('stroke_team')
        for name, group in grouped:
            stats = dict()
            stats['admissions'] = len(group)
            for field in self.fields_for_average_statistics.keys():
                if self.fields_for_average_statistics[field] == 'mean':
                    stats[field] = group[field].mean()
                elif self.fields_for_average_statistics[field] == 'median':
                    stats[field] = group[field].median()
            self.hopsital_stats_4hr_arrivals[name] = stats

        # Convert to DataFrame
        self.hopsital_stats_4hr_arrivals = \
            pd.DataFrame.from_dict(self.hopsital_stats_4hr_arrivals).T.round(2)
        self.hopsital_stats_4hr_arrivals.index.name = 'stroke_team'

        # Save
        self.hopsital_stats_4hr_arrivals.to_csv('./output/hospital_stats_4hr_arrivals.csv')


    def run(self):
        self.calculate_average_statistics()
