import numpy as np
from scipy.stats import zscore
from shap import Explainer

def calculate_bipolar_interaction_impact_scores(shap_interaction_values, true_values):
    num_features = shap_interaction_values.shape[1]
    interaction_scores = np.zeros((num_features, num_features))
    
    for i in range(shap_interaction_values.shape[0]):
        for j in range(num_features):
            for k in range(num_features):
                if true_values[i] == 1 and shap_interaction_values[i, j, k] > 0:
                    interaction_scores[j, k] += shap_interaction_values[i, j, k]
                elif true_values[i] == 0 and shap_interaction_values[i, j, k] < 0:
                    interaction_scores[j, k] += shap_interaction_values[i, j, k]
                elif true_values[i] == 1 and shap_interaction_values[i, j, k] < 0:
                    interaction_scores[j, k] += -abs(shap_interaction_values[i, j, k])
                elif true_values[i] == 0 and shap_interaction_values[i, j, k] > 0:
                    interaction_scores[j, k] += -abs(shap_interaction_values[i, j, k])

    standardized_scores = zscore(interaction_scores, axis=None)
    bipolar_normalized_scores = standardized_scores.reshape(num_features, num_features)

    return bipolar_normalized_scores

def calculate_bipolar_shap_scores_corrected(shap_values, true_values):
    num_instances, num_features = shap_values.shape
    shap_scores = np.zeros_like(shap_values)

    for i in range(num_instances):
        for j in range(num_features):
            if true_values[i] == 1 and shap_values[i, j] > 0:
                shap_scores[i, j] = shap_values[i, j]
            elif true_values[i] == 0 and shap_values[i, j] < 0:
                shap_scores[i, j] = shap_values[i, j]
            elif true_values[i] == 1 and shap_values[i, j] < 0:
                shap_scores[i, j] = -abs(shap_values[i, j])
            elif true_values[i] == 0 and shap_values[i, j] > 0:
                shap_scores[i, j] = -abs(shap_values[i, j])

    standardized_scores = zscore(shap_scores, axis=0)
    max_abs_scores = np.max(np.abs(standardized_scores), axis=0)
    bipolar_normalized_scores = standardized_scores / max_abs_scores[np.newaxis, :]

    return bipolar_normalized_scores

def convert_interaction_matrix_to_dict(interaction_matrix, bipolar_scores, feature_names):
    edge_attributes = {}
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if interaction_matrix[i, j] != 0:
                edge_attributes[(feature_names[i], feature_names[j])] = {
                    'strength': interaction_matrix[i, j],
                    'bipolar_score': bipolar_scores[i, j]
                }
    return edge_attributes
