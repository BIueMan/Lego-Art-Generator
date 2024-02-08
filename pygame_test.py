import pygame
import sys
import os
import numpy as np
from sklearn.cluster import KMeans
from clustering import get_kmean_color, cluster_points

def create_screen(image_rect):
    # Set up display
    window_width = image_rect.width + 200
    window_height = 2*image_rect.height
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Point Selector")
    return screen

def create_buttons(image_rect, k, init_button_color = None):
    # Create font for button labels
    font = pygame.font.Font(None, 36)

    # Create buttons
    buttons = []
    button_height = 40
    for i in range(k+1):
        button_rect = pygame.Rect(image_rect.width + 10, 50 + i * (button_height + 10), 50, button_height)
        if i < k:
            color = (0,255,0) if not init_button_color else init_button_color['color'][i]
            loc = None if not init_button_color else init_button_color['loc'][i]
            buttons.append({'rect':button_rect, 'font':font.render(str(i), True, (0, 0, 0)), 'color': color, 'loc': loc})
        else:
            buttons.append({'rect':button_rect, 'font':font.render('+', True, (0, 0, 0)), 'color': (0,255,0), 'loc': None})

    return buttons

def get_pixel_color(image, pos):
    # Get the color of the pixel at the specified position
    color = image.get_at(pos)
    return color

def run_app(image, clustered_pygame, original_array, image_rect, screen, buttons):
    # List to store selected points
    cluster_rect = clustered_pygame.get_rect()
    cluster_rect[1] = cluster_rect[3]

    running = True
    selected_butten = 0
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left mouse button clicked
                mouse_pos = event.pos

                # Check if any button is clicked
                for num, button in enumerate(buttons):
                    if button['rect'].collidepoint(mouse_pos):
                        selected_butten = num

                # Check if clicked on the image area
                if image_rect.collidepoint(mouse_pos):
                    # Add point to the list
                    buttons[selected_butten]['loc'] = mouse_pos
                    buttons[selected_butten]['color'] = get_pixel_color(image, mouse_pos)[:3]
                    if selected_butten == len(buttons) -1:
                        button_height = buttons[-1]['rect'][3]
                        button_rect = pygame.Rect(image_rect.width + 10, 50 + len(buttons) * (button_height + 10), 50, button_height)
                        buttons.append({'rect':button_rect, 'font':font.render('+', True, (0, 0, 0)), 'color': (0,255,0), 'loc': None})


                    # Update pixeletad lenna
                    color_list = np.array([button['color'] for button in buttons[:-1]])
                    cluster_labels = cluster_points(original_array.reshape(-1, 3), color_list)
                    clustered_array = color_list[cluster_labels].reshape(original_array.shape)
                    image_bytes = np.ascontiguousarray(clustered_array.astype(np.uint8)).tobytes()
                    clustered_pygame = pygame.image.frombuffer(image_bytes, clustered_array.shape[:2], "RGB")
                    clustered_pygame = pygame.transform.scale(clustered_pygame, image.get_size())
                    cluster_rect = clustered_pygame.get_rect()
                    cluster_rect[1] = cluster_rect[3]
                    print(f'num: {selected_butten} = {mouse_pos}, color - {buttons[selected_butten]["color"]}')

        screen.blit(image, image_rect)
        screen.blit(clustered_pygame, cluster_rect)

        # Draw red dots at selected points
        font = pygame.font.Font(None, 24)
        for num, button in enumerate(buttons):
            if buttons[num]['loc'] != None:
                pygame.draw.circle(screen, (255, 0, 0), buttons[num]['loc'], 5)

                text_surface = font.render(str(num), True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(button['loc'][0] + 15, button['loc'][1] - 15))
                screen.blit(text_surface, text_rect)

        # Draw buttons
        font = pygame.font.Font(None, 24)
        for num, button in enumerate(buttons):
            pygame.draw.rect(screen, button['color'], button['rect'])
            if selected_butten == num:
                pygame.draw.rect(screen, (255, 0, 0), button['rect'], 3)
            screen.blit(button['font'], button['rect'].center)
            text_surface = font.render(f"({button['color'][0]}, {button['color'][1]}, {button['color'][2]})", True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(button['rect'][0]+button['rect'][2]+60, button['rect'][1]+round(button['rect'][3]/2)))
            screen.blit(text_surface, text_rect)


        pygame.display.flip()

    pygame.quit()
    sys.exit()

def main():
    # first get init kmean cluster from the image
    import numpy as np
    from PIL import Image
    from clustering import get_kmean_color, cluster_points
    import cv2

    image_path = "Data/Lenna.png"
    original_image = Image.open(image_path)
    original_array = np.array(original_image.resize([16*4, 16*4]))
    original_image = np.array(original_image)
    k = 7

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
    image_path = os.path.join("Data", "Lenna.png")
    image = pygame.image.load(image_path)
    image_rect = image.get_rect()
    clustered_pygame = pygame.transform.scale(clustered_pygame, image.get_size())

    # Create screen
    screen = create_screen(image_rect)

    # Create buttons
    buttons = create_buttons(image_rect, k, init_button_color)

    # Run the app
    run_app(image, clustered_pygame, original_array, image_rect, screen, buttons)

if __name__ == "__main__":
    main()
