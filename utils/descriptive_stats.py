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
        self.full_data = pd.read_csv(
            './data/data_for_models.csv', low_memory=False)
        
        # Restrict years
        mask = ((self.full_data['year'] >= self.year_min) &
                (self.full_data['year'] <= self.year_max))
        self.full_data = self.full_data[mask]
