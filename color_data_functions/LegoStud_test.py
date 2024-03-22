import pandas as pd
from generate_pdf_func.run_app import *

if __name__ == "__main__":
    # Read Excel file
    excel_path = 'Data/ColorStud/55-lego-colors.xlsx'
    df = pd.read_excel(excel_path)

    # Extract color names and hex values
    color_dict = {}
    for index, row in df.iterrows():
        color_name = row['Color Name']
        hex_value = row['RGB (Hex)']
        color_dict[color_name] = hex_value

    print(color_dict)

    # first get init kmean cluster from the image
    import numpy as np
    from PIL import Image

    image_path = "Data/Lenna.png"
    original_image = Image.open(image_path)
    original_array = np.array(original_image.resize([16*4, 16*4]))
    original_image = np.array(original_image)

    def hex_list_to_rgb(hex_list):
        rgb_list = []
        for hex_color in hex_list:
            # Remove the '#' if present
            hex_color = hex_color.lstrip('#')
            # Convert hex to RGB
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgb_list.append(rgb)
        return rgb_list
    color_list = np.array(hex_list_to_rgb(list(color_dict.values())))
    color_list = np.array([[224, 157, 138],
                            [101,  30,  69],
                            [199,  93,  96],
                            [212, 126, 116],
                            [162,  63,  79],
                            [131,  74, 115],
                            [231, 195, 172]])

    cluster_labels = cluster_points(original_array.reshape(-1, 3), color_list)
    pixel_image = color_list[cluster_labels].reshape(original_array.shape)
    image_bytes = np.ascontiguousarray(pixel_image.astype(np.uint8)).tobytes()
    clustered_pygame = pygame.image.frombuffer(image_bytes, pixel_image.shape[:2], "RGB")

    init_button_color = {'color': [], 'loc': []}
    for _, cluster_color in enumerate(color_list):
        image_err = np.linalg.norm(original_image-cluster_color, axis=-1)

        loc = np.unravel_index(np.argmin(image_err), image_err.shape)
        init_button_color['loc'].append(loc)
        init_button_color['color'].append(original_image[loc])

    pygame.init()

    # Input image
    image_path = os.path.join("Data", "Lenna.png")
    pygame_image = pygame.image.load(image_path)
    clustered_pygame = pygame.transform.scale(clustered_pygame, pygame_image.get_size())

    # Run the app
    color_list, cluster_labels = run_app(pygame_image, clustered_pygame, original_array, init_button_color)
    for idx in range(color_list.shape[0]):
        text = f'color - {color_list[idx]}, studs - {np.sum(cluster_labels == idx)}'
        r, g, b = color_list[idx].tolist()
        print(f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")

