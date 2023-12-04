import matplotlib.pyplot as plt
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

