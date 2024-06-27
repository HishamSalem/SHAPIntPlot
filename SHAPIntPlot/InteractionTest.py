import numpy as np
import pandas as pd
import shap

class SHAPInteractionSignificance:
    def __init__(self, model, data, shap_interaction_values):
        """
        Initialize the SHAPInteractionSignificance class.

        Parameters:
        model: Trained model object.
        data: DataFrame containing the dataset.
        shap_interaction_values: Precomputed SHAP interaction values array.
        """
        self.model = model
        self.data = data
        self.shap_interaction_values = shap_interaction_values
        self.explainer = shap.TreeExplainer(model)
        self.data_columns = data.columns.tolist()
        
    def calculate_original_interaction_value(self, feature1, feature2):
        """
        Calculate the original mean SHAP interaction value for two features.

        Parameters:
        feature1: Name of the first feature.
        feature2: Name of the second feature.

        Returns:
        Mean SHAP interaction value for the two features.
        """
        index1, index2 = self.data_columns.index(feature1), self.data_columns.index(feature2)
        interaction_value = self.shap_interaction_values[:, index1, index2].mean()
        return interaction_value
    
    def permute_and_calculate(self, feature1, feature2, n_permutations=1000):
        """
        Permute one feature and calculate interaction values for permuted data.

        Parameters:
        feature1: Name of the feature to permute.
        feature2: Name of the second feature.
        n_permutations: Number of permutations to perform (default is 1000).

        Returns:
        List of permuted SHAP interaction values.
        """
        permuted_interaction_values = []

        for _ in range(n_permutations):
            data_permuted = self.data.copy()
            data_permuted[feature1] = np.random.permutation(data_permuted[feature1])
            
            permuted_shap_interaction_values = self.explainer.shap_interaction_values(data_permuted)
            index1, index2 = self.data.columns.get_loc(feature1), self.data.columns.get_loc(feature2)
            permuted_interaction_value = permuted_shap_interaction_values[:, index1, index2].mean()
            
            permuted_interaction_values.append(permuted_interaction_value)
        
        return permuted_interaction_values
    
    def test_significance(self, interaction_df, n_permutations=1000, alpha=0.05):
        """
        Test if the interaction between features is statistically significant.

        Parameters:
        interaction_df: DataFrame containing feature pairs and their interaction importance.
        n_permutations: Number of permutations to perform (default is 1000).
        alpha: Significance level (default is 0.05).

        Returns:
        DataFrame with original interaction value, p-value, and significance result for each feature pair.
        """
        results = []

        for _, row in interaction_df.iterrows():
            feature1 = row['Feature 1']
            feature2 = row['Feature 2']
            
            original_interaction_value = self.calculate_original_interaction_value(feature1, feature2)
            permuted_interaction_values = self.permute_and_calculate(feature1, feature2, n_permutations)
            
            p_value = np.mean(np.abs(permuted_interaction_values) >= np.abs(original_interaction_value))
            
            result = {
                'Feature 1': feature1,
                'Feature 2': feature2,
                'Original Interaction Value': original_interaction_value,
                'p-value': p_value,
                'Significant': p_value < alpha
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
