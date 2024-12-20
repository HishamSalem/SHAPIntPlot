# -*- coding: utf-8 -*-
"""Deployment Feature Interaction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F8gqRORri5bRjwh9m6V4K26zSc-2gCIc
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install shap
# !pip install optbinning
# !pip install SALib
# !pip install -U pyartemis
#

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
from tqdm import tqdm
import statsmodels.stats.multitest as smm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

class SHAPInteractionSignificance:
    def __init__(self, model, data, random_state=None):
        self.model = model
        self.data = data
        self.explainer = shap.TreeExplainer(model)
        self.shap_interaction_values = self.explainer.shap_interaction_values(data)
        self.data_columns = data.columns.tolist()
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def calculate_original_interaction_value(self, feature1, feature2):
        index1, index2 = self.data_columns.index(feature1), self.data_columns.index(feature2)
        interaction_value = self.shap_interaction_values[:, index1, index2].mean()
        return interaction_value

    def permute_and_calculate(self, feature1, feature2, n_permutations=1000):
        index1, index2 = self.data_columns.index(feature1), self.data_columns.index(feature2)

        def permute_once():
            data_permuted = self.data.copy()
            data_permuted[feature1] = np.random.permutation(data_permuted[feature1])
            permuted_interaction_value = self.shap_interaction_values[:, index1, index2][np.random.permutation(len(self.data))].mean()
            return permuted_interaction_value

        permuted_interaction_values = Parallel(n_jobs=-1)(delayed(permute_once)() for _ in range(n_permutations))
        return permuted_interaction_values


    def test_significance(self, interaction_df, n_permutations=1000, alpha=0.05, correction_method='fdr_bh'):
        results = []
        for _, row in tqdm(interaction_df.iterrows(), total=interaction_df.shape[0], desc="Testing significance"):
            feature1 = row['Feature1']
            feature2 = row['Feature2']
            original_interaction_value = self.calculate_original_interaction_value(feature1, feature2)
            permuted_interaction_values = self.permute_and_calculate(feature1, feature2, n_permutations)
            p_value = np.mean(np.abs(permuted_interaction_values) >= np.abs(original_interaction_value))
            result = {
                'Feature1': feature1,
                'Feature2': feature2,
                'Original Interaction Value': original_interaction_value,
                'p-value': p_value,
            }
            results.append(result)

        results_df = pd.DataFrame(results)

        # Apply multiple testing correction
        _, corrected_p_values, _, _ = smm.multipletests(results_df['p-value'], alpha=alpha, method=correction_method)
        results_df['Adjusted p-value'] = corrected_p_values
        results_df['Significant'] = results_df['Adjusted p-value'] < alpha

        return results_df

    def plot_null_distribution(self, feature1, feature2, n_permutations=1000):
        original_value = self.calculate_original_interaction_value(feature1, feature2)
        permuted_values = self.permute_and_calculate(feature1, feature2, n_permutations)

        plt.figure(figsize=(10, 6))
        sns.histplot(permuted_values, kde=True)
        plt.axvline(original_value, color='r', linestyle='--', label='Observed Value')
        plt.title(f'Null Distribution of SHAP Interaction Values for {feature1} and {feature2}')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_top_interactions(self, results_df, top_n=10):
        top_interactions = results_df.nlargest(top_n, 'Original Interaction Value')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Original Interaction Value', y='Feature1 + Feature2', data=top_interactions)
        plt.title(f'Top {top_n} Feature Interactions by SHAP Interaction Value')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Feature Pair')
        plt.tight_layout()
        plt.show()

from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import numpy as np
import pandas as pd

def model_output(params, xg_cl_final, feature_columns):
    df_sample = pd.DataFrame([params], columns=feature_columns)
    predictions = xg_cl_final.predict(df_sample)
    return float(predictions[0])

def sensitivity_analysis(X_train, y_train, model, SampleTest):
    # Define the problem
    bounds = []
    for col in X_train.columns:
        min_val = X_train[col].min()
        max_val = X_train[col].max()

        # Add a small epsilon if min and max are equal
        if min_val == max_val:
            max_val = min_val + 1e-10

        bounds.append([min_val, max_val])

    problem = {
        'num_vars': X_train.shape[1],
        'names': list(X_train.columns),
        'bounds': bounds
    }

    # Generate samples using Sobol sampler
    param_values = sobol.sample(problem, N=SampleTest)

    # Evaluate the model for each sample
    Y = np.zeros(param_values.shape[0])
    for i in range(param_values.shape[0]):
        Y[i] = model_output(param_values[i], model, X_train.columns)

    # Perform sensitivity analysis
    s1 = sobol_analyze.analyze(problem, Y)

    return s1

from optbinning import OptimalBinning, OptimalBinning2D, BinningProcess

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from optbinning import OptimalBinning, OptimalBinning2D, BinningProcess

class FeatureBinningAnalyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = None

    def get_feature_iv(self, feature_name):
        optb = OptimalBinning(name=feature_name, solver="cp")
        optb.fit(self.X[feature_name], self.y)
        binning_table = optb.binning_table.build()  # Build the table
        return binning_table['IV'].max()

    def get_2d_feature_iv(self, feature1, feature2):
        optb = OptimalBinning2D(name_x=feature1, name_y=feature2, solver="cp")
        optb.fit(self.X[feature1].values, self.X[feature2].values, self.y)
        binning_table = optb.binning_table.build()
        total_iv = binning_table['IV'].max()
        return total_iv, binning_table



    def analyze_feature_combinations(self, feature_pairs):
        results = []

        for feat1, feat2 in feature_pairs:
            # Individual IVs
            iv1 = self.get_feature_iv(feat1)
            iv2 = self.get_feature_iv(feat2)

            # 2D IV
            iv_2d, binning_table = self.get_2d_feature_iv(feat1, feat2)

            # Calculate uplift
            sum_iv = iv1 + iv2

            uplift = iv_2d - sum_iv

            results.append({
                'Feature1': feat1,
                'Feature2': feat2,
                'IV_1': iv1,
                'IV_2': iv2,
                'IV_2D': iv_2d,
                'Uplift': uplift,
                'Binning_Table': binning_table
            })

        self.results = pd.DataFrame(results)
        return self.results

    def get_top_combinations(self):
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")

        # Sort all results by uplift in descending order
        sorted_results = self.results.sort_values('Uplift', ascending=False)

        # Return relevant columns for better readability
        return sorted_results[['Feature1', 'Feature2', 'IV_1', 'IV_2', 'IV_2D', 'Uplift']]


    def get_binning_details(self, feature1, feature2):
        if self.results is None:
            raise ValueError("Run analyze_feature_combinations first")
        mask = (self.results['Feature1'] == feature1) & (self.results['Feature2'] == feature2)
        return self.results[mask]['Binning_Table'].iloc[0]

# Example usage
if __name__ == "__main__":
    # Load data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Initialize SHAPInteractionSignificance
    shap_significance = SHAPInteractionSignificance(model, X_test, random_state=42)

    # Create interaction pairs
    feature_pairs = [(X.columns[i], X.columns[j]) for i in range(len(X.columns)) for j in range(i+1, len(X.columns))]
    interaction_df = pd.DataFrame(feature_pairs, columns=['Feature1', 'Feature2'])

    # Test significance
    results = shap_significance.test_significance(interaction_df, n_permutations=100, alpha=0.05, correction_method='fdr_bh')
    # Plot top interactions
    results['Feature1 + Feature2'] = results['Feature1'] + ' + ' + results['Feature2']
    shap_significance.plot_top_interactions(results, top_n=10)

SI = sensitivity_analysis(X_train, y_train, model,SampleTest=64)

Univariate,AggregateInteractions,Pairwise=SI.to_df()

# Convert tuple index directly to columns
Pairwise['Feature1'] = [x[0] for x in Pairwise.index]
Pairwise['Feature2'] = [x[1] for x in Pairwise.index]

results

Pairwise

Pairwise.columns

from itertools import combinations

# Get all possible feature pairs
all_features = X.columns
feature_pairs = list(combinations(all_features, 2))

# Initialize analyzer
analyzer = FeatureBinningAnalyzer(X_train, y_train)

# Run analysis on all pairs
resultsIV = analyzer.analyze_feature_combinations(feature_pairs)

# Get top 10 combinations with highest uplift
top_results = analyzer.get_top_combinations()
top_results.loc[top_results['Uplift']>0.02]['Uplift'].describe()
top_results['Uplift'].describe()

import pandas as pd
import numpy as np

# Process IV method
def process_iv(resultsIV):
    iv_votes = resultsIV[['Feature1', 'Feature2']].copy()
    iv_votes['IV_Vote'] = (resultsIV['Uplift'] > 0.02).astype(int)
    return iv_votes

# Process Statistical method
def process_statistical(results):
    stat_votes = results[['Feature1', 'Feature2']].copy()
    stat_votes['Statistical_Vote'] = (results['p-value'] < 0.05).astype(int)
    return stat_votes

# Process Sensitivity Analysis method
def process_sensitivity(pairwise):
    sens_votes = pairwise[['Feature1', 'Feature2']].copy()
    threshold = pairwise['S2'].quantile(0.75)
    sens_votes['Sensitivity_Vote'] = ((pairwise['S2'] > threshold) &
                                    (pairwise['S2'] > 0)).astype(int)
    return sens_votes

# Combine all votes
def create_voting_table(resultsIV, results, pairwise):
    # Process each method
    iv_votes = process_iv(resultsIV)
    stat_votes = process_statistical(results)
    sens_votes = process_sensitivity(pairwise)

    # Merge all results
    voting_table = iv_votes.merge(
        stat_votes,
        on=['Feature1', 'Feature2'],
        how='outer'
    ).merge(
        sens_votes,
        on=['Feature1', 'Feature2'],
        how='outer'
    )

    # Fill NaN with 0
    voting_table = voting_table.fillna(0)

    # Calculate total votes
    voting_table['Total_Votes'] = (voting_table['IV_Vote'] +
                                  voting_table['Statistical_Vote'] +
                                  voting_table['Sensitivity_Vote'])

    return voting_table

# Example usage
voting_table = create_voting_table(resultsIV, results, Pairwise)

voting_table['Total_Votes'].value_counts()

voting_table

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
from tqdm import tqdm
import statsmodels.stats.multitest as smm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

class SHAPInteractionSignificance:
    def __init__(self, model, data, random_state=None):
        self.model = model
        self.data = data
        self.explainer = shap.TreeExplainer(model)
        self.shap_interaction_values = self.explainer.shap_interaction_values(data)
        self.data_columns = data.columns.tolist()
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def calculate_original_interaction_value(self, feature1, feature2):
        index1, index2 = self.data_columns.index(feature1), self.data_columns.index(feature2)
        interaction_value = self.shap_interaction_values[:, index1, index2].mean()
        return interaction_value

    def permute_and_calculate(self, feature1, feature2, n_permutations=1000):
        index1, index2 = self.data_columns.index(feature1), self.data_columns.index(feature2)

        def permute_once():
            data_permuted = self.data.copy()
            data_permuted[feature1] = np.random.permutation(data_permuted[feature1])

            # Recompute SHAP interaction values for permuted data
            shap_interaction_values_permuted = self.explainer.shap_interaction_values(data_permuted)

            # Calculate the interaction value for the permuted data
            permuted_interaction_value = shap_interaction_values_permuted[:, index1, index2].mean()

            return permuted_interaction_value

        permuted_interaction_values = Parallel(n_jobs=-1)(
            delayed(permute_once)() for _ in range(n_permutations)
        )

        return permuted_interaction_values


    def test_significance(self, interaction_df, n_permutations=1000, alpha=0.05, correction_method='fdr_bh'):
        results = []
        for _, row in tqdm(interaction_df.iterrows(), total=interaction_df.shape[0], desc="Testing significance"):
            feature1 = row['Feature1']
            feature2 = row['Feature2']
            original_interaction_value = self.calculate_original_interaction_value(feature1, feature2)
            permuted_interaction_values = self.permute_and_calculate(feature1, feature2, n_permutations)
            p_value = np.mean(np.abs(permuted_interaction_values) >= np.abs(original_interaction_value))
            result = {
                'Feature1': feature1,
                'Feature2': feature2,
                'Original Interaction Value': original_interaction_value,
                'p-value': p_value,
            }
            results.append(result)

        results_df = pd.DataFrame(results)

        # Apply multiple testing correction
        _, corrected_p_values, _, _ = smm.multipletests(results_df['p-value'], alpha=alpha, method=correction_method)
        results_df['Adjusted p-value'] = corrected_p_values
        results_df['Significant'] = results_df['Adjusted p-value'] < alpha

        return results_df

    def plot_null_distribution(self, feature1, feature2, n_permutations=1000):
        original_value = self.calculate_original_interaction_value(feature1, feature2)
        permuted_values = self.permute_and_calculate(feature1, feature2, n_permutations)

        plt.figure(figsize=(10, 6))
        sns.histplot(permuted_values, kde=True)
        plt.axvline(original_value, color='r', linestyle='--', label='Observed Value')
        plt.title(f'Null Distribution of SHAP Interaction Values for {feature1} and {feature2}')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    def plot_top_interactions(self, results_df, top_n=10):
        top_interactions = results_df.nlargest(top_n, 'Original Interaction Value')

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Original Interaction Value', y='Feature1 + Feature2', data=top_interactions)
        plt.title(f'Top {top_n} Feature Interactions by SHAP Interaction Value')
        plt.xlabel('SHAP Interaction Value')
        plt.ylabel('Feature Pair')
        plt.tight_layout()
        plt.show()

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate independent features
X = pd.DataFrame({
    'Feature_A': np.random.normal(0, 1, n_samples),
    'Feature_B': np.random.normal(0, 1, n_samples),
    'Feature_C': np.random.normal(0, 1, n_samples),
    'Feature_D': np.random.normal(0, 1, n_samples)
})

# Introduce explicit interaction between Feature_A and Feature_B
interaction_term = X['Feature_A'] * X['Feature_B']

# Define the target variable influenced by the interaction term
# For classification, we'll apply a logistic function
logits = 0.5 * X['Feature_A'] + 0.5 * X['Feature_B'] + 2 * interaction_term
probabilities = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probabilities)

# Combine features and target into a DataFrame
data = X.copy()
data['Target'] = y

# Save the synthetic data to a CSV file (optional)
# data.to_csv('synthetic_data.csv', index=False)

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load the synthetic data (if saved to CSV)
# data = pd.read_csv('synthetic_data.csv')

# Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

import pandas as pd


# Initialize the SHAPInteractionSignificance object
shap_interaction = SHAPInteractionSignificance(model, X_test, random_state=42)

# Prepare a DataFrame of feature pairs
features = X_test.columns.tolist()
interaction_pairs = [(f1, f2) for i, f1 in enumerate(features) for f2 in features[i+1:]]

interaction_df = pd.DataFrame(interaction_pairs, columns=['Feature1', 'Feature2'])

# Test significance of interactions
results_df = shap_interaction.test_significance(interaction_df, n_permutations=1000, alpha=0.05)

# Display significant interactions
significant_interactions = results_df[results_df['Significant']]
print(significant_interactions)

results_df



from artemis.interactions_methods.model_agnostic import GreenwellMethod

# Initialize the Greenwell method
greenwell_method = GreenwellMethod()

# Fit the method using the trained model and test data
greenwell_method.fit(model, X=X_test, show_progress=True)

# Access the interaction strengths
interaction_strengths = greenwell_method.ovo

# Convert to DataFrame
interaction_strengths_df = interaction_strengths.reset_index().rename(columns={'index': 'Interaction'})

# Display top interactions
top_interactions = interaction_strengths_df.sort_values(by='Greenwell Variable Interaction Measure', ascending=False)
print(top_interactions.head(10))

# Prepare your method's interactions
your_method_interactions = significant_interactions[['Feature1', 'Feature2']].copy()
your_method_interactions['Pair'] = your_method_interactions['Feature1'] + '_' + your_method_interactions['Feature2']

# Prepare Greenwell method's interactions
greenwell_interactions = top_interactions.copy()
greenwell_interactions['Pair'] = greenwell_interactions['Variable1'] + '_' + greenwell_interactions['Variable2']

# Merge the results on the 'Pair' column
merged_results = pd.merge(
    your_method_interactions,
    greenwell_interactions,
    on='Pair',
    how='inner'
)

# Display the merged results
print(merged_results[['Feature1', 'Feature2', 'strength']])