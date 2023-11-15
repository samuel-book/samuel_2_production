import pandas as pd

class ThrombolysisChoiceModel:
    """
    Predicts outcome

    Attributes:

        

    Methods:
        
    """
    
    def __init__(self):
        """
        """

    # Load data
    try:
        self.data = pd.read_csv(
            './data/data_for_ml/complete_ml_data.csv', low_memory=False)
    except:
        print ("./data/data_for_ml/complete_ml_data.csv does not exist")
    