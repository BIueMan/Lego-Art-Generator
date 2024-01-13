from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def repaint_image(original_image, output_path):
    # Load the image
    original_array = np.array(original_image)
    height, width, _ = original_array.shape

    # Reshape the array for clustering
    reshaped_array = original_array.reshape((height * width, 3))

    # Apply K-means clustering
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_array)

    # Get cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_.astype(int)

    # Assign each pixel to the closest cluster center
    labels = kmeans.predict(reshaped_array)
    clustered_array = cluster_centers[labels].reshape((height, width, 3))

    # Create a new image with the clustered colors
    clustered_image = Image.fromarray(np.uint8(clustered_array))
    
    # Save the clustered image
    clustered_image.save(output_path)

    # Display the original and clustered images
    scale = 16*3*10
    original_image.resize([scale,scale]).show(title="Original Image")
    clustered_image.resize([scale, scale]).show(title="Clustered Image")
    # Plot the cluster centers, color assignments, and every point with its color
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Cluster Centers")
    plt.imshow(cluster_centers.reshape((1, k, 3)))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Color Assignments")
    plt.scatter(reshaped_array[:, 0], reshaped_array[:, 1], c=labels, cmap='viridis', s=5)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=100)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Every Point with Its Color")
    plt.scatter(reshaped_array[:, 0], reshaped_array[:, 1], c=reshaped_array/255, s=5)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    # Provide the path to your image and the desired output path
    image_path = "Data/Lenna.png"
    output_path = "Data/Lenna_out.png"

    original_image = Image.open(image_path)
    original_image = original_image.resize([16*3, 16*3])

    repaint_image(original_image, output_path)
