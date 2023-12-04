import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class StrokeTeamReporting():
    """
    Utilities to produce output for a specific stroke team.
    """

    def __init__(self, team_name):
        """
        Initialise
        """
        self.team_name = team_name
        self.import_data()


    def import_data(self):
        """
        Import data for the stroke team
        """

        try:
            self.sim_results = pd.read_csv('./output/sim_results_all.csv')
            mask = self.sim_results['stroke_team'] == self.team_name
            self.sim_results = self.sim_results[mask]
        except:
            raise ValueError('No simulation results found. Run simulation first.')
        
        
    def plot_improvement(self, ax=None, **kwargs):
        """
        Plot improvement in outcomes with changes
        """

        """Plot and save overall simulation results"""

        fig = plt.figure(figsize=(10,7))

        x = list(self.sim_results['scenario'].values)
        # Replace all _ in x with + for plotting
        x = [i.replace('_', '+') for i in x]

        ax1 = fig.add_subplot(121)        
        y1 = self.sim_results['Percent_Thrombolysis_(mean)'].values
        ax1.bar(x,y1)
        ax1.set_ylim(0,20)
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0,22,2))
        ax1.set_title('Thrombolysis use (%)')
        ax1.set_ylabel('Thrombolysis use (%)')
        ax1.set_xlabel('Scenario')
        ax1.grid(axis = 'y')

        ax2 = fig.add_subplot(122)
        y1 = self.sim_results['Additional_good_outcomes_per_1000_patients_(mean)'].values
        ax2.bar(x,y1, color='r')
        ax2.set_ylim(0,20)
        plt.xticks(rotation=90)
        plt.yticks(np.arange(0,22,2))
        ax2.set_title('Additional good outcomes\nper 1,000 admissions')
        ax2.set_ylabel('Additional good outcomes\nper 1,000 admissions')
        ax2.set_xlabel('Scenario')
        ax2.grid(axis = 'y')

        # Set a title above both figures
        fig.suptitle(f'Potential improvement for {self.team_name}')

        fig.tight_layout(pad=2)

        return fig