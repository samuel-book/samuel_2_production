import pandas as pd

class DataLoad:
    """
    Loads data ready for models.

    Attributes:

    full_data:
        Pandas dataframe of full SSNAP data (Cleaned)
        

    Methods:
        
    """
    
    def __init__(self, *args, **kwargs):
        """
        Creates the data load object
        """

        self.thrombolysis_choice_fields = [
            'arrival_to_scan_time',
            'infarction',
            'stroke_severity',
            'precise_onset_known',
            'prior_disability',
            'stroke_team',
            'afib_anticoagulant',
            'onset_to_arrival_time',
            'onset_during_sleep',
            'age'
        ]
        
        # Overwrite any data paramters

        for dictionary in args:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Load stored data

        self.load_data()

    def load_data(self):
        """
        Load data and create datasets
        """

        # Load full data
        self.full_data = pd.read_csv(
            './data/data_for_models.csv', low_memory=False)