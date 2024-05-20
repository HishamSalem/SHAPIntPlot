from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import shap
from my_shap_package import calculate_bipolar_interaction_impact_scores, calculate_bipolar_shap_scores_corrected, convert_interaction_matrix_to_dict, create_final_network_graph

# Load and split the dataset
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model and calculate SHAP values
model = XGBClassifier()
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_proba)
print(f"AUC score: {auc}")

explainer = shap.Explainer(model)
shap_values = explainer(X_train)
shap_interaction_values = explainer.shap_interaction_values(X_train)
feature_importance = np.abs(shap_values.values).mean(0)
interaction_focus = np.abs(shap_interaction_values).sum(0)

shap_bipolar = calculate_bipolar_shap_scores_corrected(shap_values.values, y_train).sum(0)
node_attributes = {feature_name: (importance, shap) for feature_name, importance, shap in zip(feature_names, feature_importance, shap_bipolar)}
interaction_scores = calculate_bipolar_interaction_impact_scores(shap_interaction_values, y_train)
edge_attributes_dict = convert_interaction_matrix_to_dict(interaction_focus, interaction_scores, feature_names)

create_final_network_graph(node_attributes, edge_attributes_dict, max_node_size=1500, max_edge_width=2, figsize=(8, 10),
                           spread=1 * len(node_attributes), text_size=8, min_node_size=100, iterations=100, 
                           scale=None, use_average_degree=True)
