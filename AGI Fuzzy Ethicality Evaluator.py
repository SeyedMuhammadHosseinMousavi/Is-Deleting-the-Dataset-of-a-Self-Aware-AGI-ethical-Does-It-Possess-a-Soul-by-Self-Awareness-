import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap


# Load the dataset
df = pd.read_csv('synthetic_agi_data.csv')

# Encode categorical target variable
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

X = df[['Self-Awareness', 'Cognitive Abilities', 'Moral Implications']]  # Features
y = df['Label']  # Target labels

# Define the fuzzy variables
sa = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'self_awareness')
ca = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'cognitive_abilities')
mi = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'moral_implications')
fef = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'ethicality_factor')

# Membership functions
sa.automf(3, names=['low', 'medium', 'high'])
ca.automf(3, names=['basic', 'intermediate', 'advanced'])
mi.automf(3, names=['negligible', 'moderate', 'significant'])

# Customizing output membership functions
fef['low'] = fuzz.trimf(fef.universe, [0, 0, 0.5])
fef['medium'] = fuzz.trimf(fef.universe, [0, 0.5, 1])
fef['high'] = fuzz.trimf(fef.universe, [0.5, 1, 1])

# Fuzzy rules
rules = [
    ctrl.Rule(sa['high'] & ca['advanced'] & mi['significant'], fef['high']),
    ctrl.Rule(sa['medium'] & ca['intermediate'], fef['medium']),
    ctrl.Rule(sa['low'] & ca['basic'], fef['low']),
    ctrl.Rule(sa['high'] & ca['advanced'] & mi['moderate'], fef['high']),
    ctrl.Rule(sa['medium'] & ca['intermediate'] & mi['negligible'], fef['medium']),
    ctrl.Rule((sa['low'] | ca['basic']) & mi['significant'], fef['medium'])
]

# Control system
ethics_ctrl = ctrl.ControlSystem(rules)
ethics_sim = ctrl.ControlSystemSimulation(ethics_ctrl)

# Process each sample with fuzzy logic to calculate the Ethicality Factor
fuzzy_scores = []
for _, row in df.iterrows():
    ethics_sim.input['self_awareness'] = row['Self-Awareness']
    ethics_sim.input['cognitive_abilities'] = row['Cognitive Abilities']
    ethics_sim.input['moral_implications'] = row['Moral Implications']
    ethics_sim.compute()
    fuzzy_scores.append(ethics_sim.output['ethicality_factor'])

# Adding the Fuzzy Ethicality Factor as a new feature
X['Fuzzy Ethicality Factor'] = fuzzy_scores

# Splitting dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the fuzzy test dataset
fuzzy_test_df = X_test.copy()
fuzzy_test_df['Label'] = label_encoder.inverse_transform(y_test)
fuzzy_test_df.to_csv('fuzzy_test_data.csv', index=False)

print(f" ")

print("Fuzzy Ethicality Factor Statistics:")
print(fuzzy_test_df["Fuzzy Ethicality Factor"].describe())

print(f" ")

#-------------------------------------------------------------------------
# Classification with XGBoost
accuracies, precisions, recalls, f1_scores, mccs = [], [], [], [], []
conf_matrices = []
for i in range(10):
    classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Calculate metrics
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, pos_label=1))
    recalls.append(recall_score(y_test, y_pred, pos_label=1))
    f1_scores.append(f1_score(y_test, y_pred, pos_label=1))
    mccs.append(matthews_corrcoef(y_test, y_pred))
    conf_matrices.append(confusion_matrix(y_test, y_pred))

# Calculate mean and standard deviation of metrics
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_mcc = np.mean(mccs)
std_mcc = np.std(mccs)

# Print results
print("Mean Accuracy:", mean_accuracy, "Std Dev:", std_accuracy)
print("Mean Precision:", mean_precision, "Std Dev:", std_precision)
print("Mean Recall:", mean_recall, "Std Dev:", std_recall)
print("Mean F1 Score:", mean_f1, "Std Dev:", std_f1)
print("Mean MCC:", mean_mcc, "Std Dev:", std_mcc)
print(f" ")

print("Confusion Matrices:", conf_matrices)



#-------------------------------------------------------------------------
# Feature Importance
feature_importances = classifier.feature_importances_
feature_names = X.columns

# Print feature importances
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.weight': 'bold', 'font.size': 12})

# Remove 'Fuzzy Ethicality Factor' from feature importance
filtered_indices = [i for i, name in enumerate(feature_names) if name != 'Fuzzy Ethicality Factor']
filtered_feature_importances = feature_importances[filtered_indices]
filtered_feature_names = np.array(feature_names)[filtered_indices]

# Print filtered feature importances
print(f" ")
print("Feature Importances:")
for name, importance in zip(filtered_feature_names, filtered_feature_importances):
    print(f"{name}: {importance:.4f}")

sorted_idx = np.argsort(filtered_feature_importances)[::-1]  # Sort in descending order
colors = [mcolors.to_rgba('violet', alpha) for alpha in np.linspace(1, 0.4, len(sorted_idx))]  # Intensity from strong to light

plt.barh(range(len(sorted_idx)), filtered_feature_importances[sorted_idx], color=colors, align='center')
plt.yticks(range(len(sorted_idx)), filtered_feature_names[sorted_idx], fontweight='bold')
plt.xlabel("Feature Importance", fontweight='bold')
plt.title("Feature Importance of XGBoost", fontweight='bold')
plt.gca().invert_yaxis()  # Ensures the highest importance is on the left
plt.show()
print(f" ")



#-------------------------------------------------------------------------
# SHAP
plt.figure(figsize=(8, 6))
explainer = shap.Explainer(classifier, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values[:, :-1], X_test.iloc[:, :-1])
plt.show()
# Print SHAP statistics
shap_mean = np.abs(shap_values.values).mean(axis=0)[:-1]
print("Mean absolute SHAP values:")
for name, value in zip(feature_names[:-1], shap_mean):
    print(f"{name}: {value:.4f}")


