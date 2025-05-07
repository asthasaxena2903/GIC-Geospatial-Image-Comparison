import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON file
with open(r"D:\project deepmatrix\ml_interns\parallel_light_glue\output_path\output.json", "r") as f:
    data = json.load(f)

# Extract similarity scores
scores = []
for key, matches in data.items():
    if isinstance(matches, list):  
        for match in matches:
            if isinstance(match, list) and len(match) == 3:
                scores.append(match[2])  # Extract similarity score

if not scores:
    print("No valid similarity scores found!")
    exit()

# Print statistics
print(f"Total Comparisons: {len(scores)}")
print(f"Min Score: {min(scores)}")
print(f"Max Score: {max(scores)}")
print(f"Mean Score: {np.mean(scores)}")

# Plot histogram
plt.hist(scores, bins=20, color='blue', edgecolor='black', alpha=0.75)
plt.xlabel('Similarity Score')
plt.ylabel('Frequency')
plt.title('Histogram of Similarity Scores')
plt.grid(True)
plt.show()
