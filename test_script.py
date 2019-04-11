from exmap import Exmap
import pandas as pd

voting_data = pd.read_csv("sample_data/voting_data.csv")
X = voting_data.drop('Clinton', axis=1)
y = voting_data['Clinton']
targets_limit = 10

X_test, test_coords, explainer, model = Exmap.train_model(features=X, target=y, task_type="clf")
explanations = Exmap.compute_lime_explanations(test_data=X_test, explainer=explainer, model=model, targets_limit=targets_limit, num_features=15, task_type="clf")
Exmap.generate_lime_plots(explanations_list=explanations, image_prefix="voting")
m = Exmap.make_map(test_coords=test_coords, targets_number=targets_limit)
m.save('voting_map.html')
