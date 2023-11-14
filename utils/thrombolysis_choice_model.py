import pandas as pd

from sklearn.model_selection import train_test_split


class ThrombolysisChoiceModel:

    """
    Predicts use of thrombolysis

    Attributes:

        

    Methods:
        
    """
    
    def __init__(self, data_loader):
        """
        Constructor for thrombolysis choice model
        """

        self.data_loader = data_loader

        # Call processing of data
        self.process_data()


    def process_data(self):
        """
        Process full SSNAP data for decision choice model
        """

        # Limit to ambulance arrivals if required
        if self.data_loader.limit_to_ambo:
            mask = self.data_loader.full_data['arrive_by_ambulance'] == 1
            self.data = self.data_loader.full_data[mask]
        else:
            self.data = self.data_loader.full_data.copy()

        # Limit to 4 hours onset to arrival
        mask = self.data_loader.full_data['onset_to_arrival_time'] <= 240
        self.data = self.data_loader.full_data[mask]

        # Limit to required fields
        required_fields = self.data_loader.thrombolysis_choice_fields
        self.data = self.data[required_fields]

        # Replace afib_anticoagulant NaN with 0
        self.data['afib_anticoagulant'].fillna(0, inplace=True)

        # Split X and Y
        X = self.data.drop('thrombolysis', axis=1)
        y = self.data['thrombolysis']

        # One hot encode stroke_team
        X_hosp = pd.get_dummies(X['stroke_team'], prefix = 'team')
        X = pd.concat([X, X_hosp], axis=1)
        X.drop('stroke_team', axis=1, inplace=True)

        # Split training and test
        strat = (self.data['stroke_team'].map(str) + '-' + 
                 self.data['thrombolysis'].map(str))
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.25, random_state=42, stratify=strat)