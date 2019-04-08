import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import lime
import lime.lime_tabular
import os
import folium
import base64
from tqdm import tqdm

class Exmap:


    def train_model(features, target, task_type):
        """
        Train a model and create and explainer object.
        """
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        test_coords = X_test[['lat', 'lng']]
        X_train = X_train.values
        X_test = X_test.values

        if task_type == "clf":
            xgb_model = xgb.XGBClassifier(objective="reg:logistic", random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            preds = xgb_model.predict(X_test)
            preds_probas = xgb_model.predict_proba(X_test)[:,1]
            print("The accuracy is: {}".format(round(metrics.accuracy_score(y_test, preds), 3)))
            print("Confusion matrix:")
            print(metrics.confusion_matrix(y_test, preds))
            explainer = lime.lime_tabular.LimeTabularExplainer(X_test, feature_names=list(features.columns), class_names=["0", "1"], discretize_continuous=True)
        elif task_type == "regr":
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            preds = xgb_model.predict(X_test)
            explainer = lime.lime_tabular.LimeTabularExplainer(X_test, feature_names=list(features.columns), discretize_continuous=True, mode="regression")

        return X_test, test_coords, explainer, xgb_model


    def compute_lime_explanations(test_data, explainer, model, targets_limit, num_features, task_type):
        """
        Generate LIME explanations.
        """
        if task_type == "clf":
            explanations = []
            for i in tqdm(range(0, targets_limit)):
                temp_explanation = explainer.explain_instance(test_data[i], model.predict_proba, num_features=num_features)
                explanation = temp_explanation.as_list(1)
                explanations.append(explanation)
        elif task_type == "regr":
            explanations = []
            for i in tqdm(range(0, targets_limit)):
                temp_explanation = explainer.explain_instance(test_data[i], model.predict, num_features=num_features)
                explanation = temp_explanation.as_list(1)
                explanations.append(explanation)

        return explanations

    def generate_lime_plots(explanations_list, image_prefix):
        """
        Create explanation plots.
        """
        for i in tqdm(range(0, len(explanations_list))):
            df = pd.DataFrame(explanations_list[i])
            df.columns = ['rule', 'weight']
            ax = df.plot(kind='bar', y='weight', x='rule')
            fig = ax.get_figure()
            fig.savefig(f"images/{image_prefix}_output_{i}.jpeg", bbox_inches = 'tight')

    def make_map(test_coords, targets_number):
        """
        Build interactive Folium map.
        """
        m = folium.Map(
            location=[39, -102],
            tiles='Stamen Terrain',
            zoom_start=5
        )

        filelist = []
        for folder, subs, files in os.walk("images"):
          for filename in files:
              if filename.endswith(".jpeg"):
                filelist.append(os.path.abspath(os.path.join(folder, filename)))

        for i in range(0, targets_number):
            encoded = base64.b64encode(open(filelist[i], 'rb').read()).decode()
            html = '<img src="data:image/jpeg;base64,{}">'.format
            iframe = folium.IFrame(html(encoded), width=560, height=500)
            popup = folium.Popup(iframe, max_width=1000)
            marker = folium.Marker(location=[test_coords.iloc[i]['lat'], test_coords.iloc[i]['lng']], popup=popup, icon=folium.Icon(icon='user'))
            marker.add_to(m)

        for file in filelist:
            os.remove(file)

        return m
