from utils.data_process import DataProcess
from utils.thrombolysis_choice_model import ThrombolysisChoiceModel

create_new_data = False

# Create new data if required
if create_new_data:
    data_processor = DataProcess()
    data_processor.create_ml_data()

thrombolysis_choice_model = ThrombolysisChoiceModel()
