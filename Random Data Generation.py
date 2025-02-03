import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
num_samples = 300
# Self-Awareness (SA), Cognitive Abilities (CA), Moral Implications (MI) as continuous features from 0 to 1
data = {
    "Self-Awareness": np.random.rand(num_samples),
    "Cognitive Abilities": np.random.rand(num_samples),
    "Moral Implications": np.random.rand(num_samples)
}

df = pd.DataFrame(data)

# Define a simple rule to assign labels: if the mean of the three features is greater than 0.5, it's ethical; otherwise, unethical
df['Label'] = (df.mean(axis=1) > 0.5).replace({True: 'Ethical', False: 'Unethical'})

# Display the first few rows of the dataframe
print(df.head())

# Save the dataframe to a CSV file
df.to_csv("synthetic_agi_data.csv", index=False)
