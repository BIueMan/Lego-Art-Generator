import numpy as np
from sklearn.cluster import KMeans

# Define the input colors and studs
colors = np.array([
    [205, 133, 125], [122, 37, 68], [174, 67, 79], [243, 201, 153],
    [129, 75, 125], [214, 160, 150], [205, 95, 95], [226, 131, 113],
    [94, 23, 64], [102, 38, 83], [149, 54, 75], [232, 201, 186],
    [191, 84, 89], [159, 81, 99], [185, 112, 117], [218, 111, 104],
    [147, 104, 144], [224, 182, 168], [235, 158, 125], [120, 56, 100]
])

# Convert colors to float and scale to [0,1]
colors = colors.astype(float) / 255.0

# Define the number of clusters
num_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(colors)

# Initialize a dictionary to store groups of colors
color_groups = {i: [] for i in range(num_clusters)}

# Group colors by their cluster
for i, color in enumerate(colors):
    cluster_label = clusters[i]
    color_groups[cluster_label].append((color * 255).astype(int))

# Print color groups
for cluster_label, group in color_groups.items():
    print(f"Cluster {cluster_label}:")
    for color in group:
        text = f'color - {color}'
        r, g, b = color
        print(f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
    print()
