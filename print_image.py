import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_text_color(rgb_color):
    # Calculate luminance to determine if the color is light or dark
    luminance = (0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2]) / 255
    if luminance > 0.5:
        return (0, 0, 0)  # Black text for light colors
    else:
        return (255, 255, 255)  # White text for dark colors

def create_image(label_matrix, rgb_colors, circle_size=5, background_color=(0, 0, 0), font_path=None):
    width = label_matrix.shape[1] * circle_size
    height = label_matrix.shape[0] * circle_size
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    
    # Load font if specified
    font = None
    if font_path:
        font = ImageFont.truetype(font_path, int(circle_size*0.8))

    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            label = label_matrix[i, j]
            color = tuple(rgb_colors[label])
            text_color = get_text_color(color)
            x = j * circle_size
            y = i * circle_size
            draw.ellipse([x, y, x + circle_size, y + circle_size], fill=color)
            # Get label text
            label_text = str(label)
            # Get label text bbox
            text_bbox = draw.textbbox((x, y), label_text, font=font)
            # Calculate text position
            text_x = x + (circle_size - (text_bbox[2] - text_bbox[0])) / 2
            text_y = y + (circle_size - (text_bbox[3] - text_bbox[1])) / 2
            # Draw label text
            draw.text((text_x, text_y), label_text, fill=text_color, font=font)

    return image

if __name__ == "__main__":
    # Example usage:
    label_matrix = np.array([[0, 1, 2],
                            [3, 4, 5],
                            [6, 7, 8]])

    rgb_colors = np.array([[255, 0, 0],   # Color for label 0
                        [0, 255, 0],   # Color for label 1
                        [0, 0, 255],   # Color for label 2
                        [255, 255, 0], # Color for label 3
                        [255, 0, 255], # Color for label 4
                        [0, 255, 255], # Color for label 5
                        [128, 0, 128], # Color for label 6
                        [128, 128, 0], # Color for label 7
                        [0, 128, 128]])# Color for label 8

    font_path = 'C:\Windows\Fonts\Arial.ttf'
    image = create_image(label_matrix, rgb_colors, circle_size=50, font_path = font_path)
    image.save("output.png")
