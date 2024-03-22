import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from scipy.spatial.distance import cdist

def get_kmean_color(original_image, k=5):
    # Load the image
    original_array = np.array(original_image)
    height, width, _ = original_array.shape

    # Reshape the array for clustering
    reshaped_array = original_array.reshape((height * width, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reshaped_array)

    # Get cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_.astype(int)

    # Assign each pixel to the closest cluster center
    labels = kmeans.predict(reshaped_array)
    clustered_array = cluster_centers[labels].reshape((height, width, 3))

    return cluster_centers, clustered_array

def cluster_points(points, cluster):
    distances_matrix = cdist(cluster, points, metric='euclidean')
    cluster_labels = np.argmin(distances_matrix, axis=0)
    return cluster_labels

def plot_3d_color(color_points, cluster_points):
    # Plotting the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot color points with original colors
    for i in range(len(color_points)):
        ax.scatter(color_points[i, 0], color_points[i, 1], color_points[i, 2], c=[color_points[i]/255])

    # Plot other points with different colors based on their clusters
    for i in range(len(cluster_points)):
        ax.scatter(cluster_points[i, 0], cluster_points[i, 1], cluster_points[i, 2], c=[cluster_points[i]/255], marker='x', s=100, label=f'Cluster {cluster_points[i]}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def plot_3d_cluster(color_points, cluster_points, cluster_labels):
    # Plotting the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot color points with original colors
    for i in range(len(color_points)):
        ax.scatter(color_points[i, 0], color_points[i, 1], color_points[i, 2], c=[cluster_points[cluster_labels[i]]/255])
    
    # Plot other points with different colors based on their clusters
    for i in range(len(cluster_points)):
        ax.scatter(cluster_points[i, 0], cluster_points[i, 1], cluster_points[i, 2], c=[cluster_points[i]/255], marker='x', s=100, label=f'Cluster {cluster_points[i]}')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def plot_clustered_image(original_image, cluster_labels, points_to_cluster, scale = 16*3*10):
    height, width, _ = original_array.shape

    clustered_array = points_to_cluster[cluster_labels].reshape((height, width, 3))
    clustered_image = Image.fromarray(np.uint8(clustered_array))

    plot_image = Image.new('RGB', (scale * 2, scale))

    plot_image.paste(original_image.resize([scale, scale]), (0, 0))
    plot_image.paste(clustered_image.resize([scale, scale]), (scale, 0))

    plot_image.show()

if __name__ == "__main__":
    # load image and resize
    image_path = "Data/Lenna.png"
    output_path = "Data/Lenna_out.png"
    original_image = Image.open(image_path)
    original_array = np.array(original_image.resize([16*3, 16*3]))
    height, width, _ = original_array.shape

    point_on_image = ([100, 300], [10, 10], [200, 200])
    
    points_to_cluster = np.array([np.array(original_image)[point[0], [point[1]]][0].tolist() for point in point_on_image])
    points_to_cluster, _ = get_kmean_color(original_image, k=7)
    # flaten image
    point_of_image = original_array.reshape((height * width, 3))
    
    cluster_labels = cluster_points(point_of_image, points_to_cluster)

    # plot_3d_color(point_of_image, points_to_cluster)
    # plot_3d_cluster(point_of_image, points_to_cluster, cluster_labels)

    plot_clustered_image(original_image, cluster_labels, points_to_cluster)
