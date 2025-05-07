
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load JSON file
file_path = r"D:\project deepmatrix\ml_interns\parallel_light_glue\output\.ipynb_checkpoints\output-checkpoint.json" # Replace with your file path
with open(file_path, "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
matches = []
for key, value in data.items():
    for match in value:
        matches.append({
            'image_pair': key,
            'coord_1': match[0],
            'coord_2': match[1],
            'match_score': match[2]
        })

df = pd.DataFrame(matches)

# Print first few rows
print("First few rows of the data:")
print(df.head())

# Perform basic analysis
print("\nSummary Statistics:")
print(df.describe())

# Plot the distribution of match scores
plt.hist(df['match_score'], bins=20, edgecolor='black')
plt.title('Distribution of Match Scores')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.show()
