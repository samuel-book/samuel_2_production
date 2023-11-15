import pandas as pd

class ThrombolysisOutcomeModel:
    """
    Predicts outcome

    Attributes:

        

    Methods:
        
    """
    
    def __init__(self):
        """
        """

        # Load data
        self.data = pd.read_csv(
                './data/data_for_ml/complete_ml_data.csv', low_memory=False)
    