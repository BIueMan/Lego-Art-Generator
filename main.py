import pygame
import sys
import os
import numpy as np
from math_function.clustering import get_kmean_color, cluster_points
from generate_pdf_func.create_pdf import create_pdf_from_directory
from generate_pdf_func.generate_pixel_images import generate_pixel_images
from generate_pdf_func.run_app import *
from generate_pdf_func.personal_touch import personal_touch
from PIL import Image


def main(image_path:str, size:list, k:int):
    # first get init kmean cluster from the image
    original_image = Image.open(image_path)
    original_image = original_image.convert('RGB')
    original_array = np.array(original_image.resize([16*size[0], 16*size[1]]))
    original_image = np.array(original_image)

    points_to_cluster, clustered_array = get_kmean_color(original_array, k)
    image_bytes = np.ascontiguousarray(clustered_array.astype(np.uint8)).tobytes()
    clustered_pygame = pygame.image.frombuffer(image_bytes, clustered_array.shape[:2], "RGB")

    init_button_color = {'color': [], 'loc': []}
    for _, cluster_color in enumerate(points_to_cluster):
        image_err = np.linalg.norm(original_image-cluster_color, axis=-1)

        loc = np.unravel_index(np.argmin(image_err), image_err.shape)
        init_button_color['loc'].append(loc)
        init_button_color['color'].append(original_image[loc])

    pygame.init()

    # Input image
    image_path = image_path # os.path.join("Data", "Lenna.png")
    pygame_image = pygame.image.load(image_path)
    pygame_image = pygame.transform.scale(pygame_image, (500, 500))
    clustered_pygame = pygame.transform.scale(clustered_pygame, pygame_image.get_size())
    # Run the app
    color_list, cluster_labels = run_app(pygame_image, clustered_pygame, original_array, init_button_color)
    
    pygame.init()
    image_path = os.path.join("Data", "Color_list_image.png")
    pygame_image = pygame.image.load(image_path)
    pygame_image = pygame.transform.scale(pygame_image, (500, 500))
    # update buttton color
    init_button_color['color'] = np.copy(color_list)
    init_button_color['loc'] = -1 * np.ones((len(color_list), 2), dtype=np.int16)
    
    original_array = color_list[cluster_labels.reshape(-1)].reshape(original_array.shape)
    image_bytes = np.ascontiguousarray(original_array.astype(np.uint8)).tobytes()
    clustered_pygame = pygame.image.frombuffer(image_bytes, original_array.shape[:2], "RGB")
    clustered_pygame = pygame.transform.scale(clustered_pygame, pygame_image.get_size())
        
    # Run the app
    color_list, cluster_labels = run_app(pygame_image, clustered_pygame, original_array, init_button_color, keep_cluster = True)
    
    color_list, cluster_labels = personal_touch(color_list, cluster_labels)
    
    # plot colors
    for idx in range(color_list.shape[0]):
        text = f'color - {color_list[idx]}, studs - {np.sum(cluster_labels == idx)}'
        r, g, b = color_list[idx].tolist()
        print(f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
    
    rows, cols = cluster_labels.shape[0], cluster_labels.shape[1]
    # Create the directory if it doesn't exist
    os.makedirs('output/', exist_ok=True)
    os.makedirs('output/image', exist_ok=True)
    # Extract a patch of 16x16
    for i, j in [(i, j) for i in range(0, rows, 16) for j in range(0, cols, 16)]:
        patch = cluster_labels[i:i+16, j:j+16]
        # Call the function to print the patch
        image = generate_pixel_images(patch, color_list, circle_size=50)
        image.save(f"output/image/sub_image_{int(i/16)}_{int(j/16)}.png")
        
    color_image = generate_pixel_images(np.array(range(color_list.shape[0])).reshape((1, -1)), color_list, circle_size=50, font_path = None, add_color_names=True)
    color_image.save(f"output/color_list.png")

    full_image = generate_pixel_images(cluster_labels, color_list, circle_size=50, font_path = None, add_number=False)
    full_image.save(f"output/full_image.png")

    # save images
    image_dict_path = "output/image"
    color_image_path = "output/color_list.png"
    output_path = "output/output.pdf"
    full_image_path = "output/full_image.png"

    create_pdf_from_directory(image_dict_path, color_image_path, output_path, full_image_path)

if __name__ == "__main__":
    image_path = "Data/pokemon.jpeg"
    size = [4, 4]
    k_mean = 9
    main(image_path, size, k_mean)
