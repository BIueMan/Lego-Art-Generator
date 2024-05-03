import pygame
import numpy as np


def personal_touch(color_matrix, label_image, scale=10):
    # Initialize Pygame
    pygame.init()

    # Get image dimensions
    label_image = np.flipud(np.rot90(label_image, 1))
    height, width = label_image.shape

    # Set up screen dimensions based on scale
    screen_width = width * scale + 200
    screen_height = height * scale
    screen = pygame.display.set_mode((screen_width, screen_height))

    # Create a numpy array of shape (height, width, 3) to store the colors
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the colored_image array with colors based on labels
    for label in range(len(color_matrix)):
        color = color_matrix[label]
        colored_image[label_image == label] = color

    # Scale up colored image based on scale
    scaled_colored_image = np.repeat(np.repeat(colored_image, scale, axis=0), scale, axis=1)

    # Create a Pygame surface from the scaled colored image
    surf = pygame.surfarray.make_surface(scaled_colored_image)

    # Blit the scaled surface to the left of the screen
    screen.blit(surf, (0, 0))

    # Add a column of squares to the right of the image
    square_size = 30
    selected_color = color_matrix[0]
    selected_label = 0
    for label, color in enumerate(color_matrix):
        pygame.draw.rect(screen, color, (width * scale + 10, label * square_size, square_size, square_size))
        font = pygame.font.SysFont(None, int(square_size / 2))
        text = font.render(str(label), True, (255, 255, 255))
        screen.blit(text, (width * scale + square_size + 20, label * square_size + 5))
        text = font.render(str(color), True, (255, 255, 255))
        screen.blit(text, (width * scale + square_size + 20, label * square_size + square_size / 2))

    # Update display
    pygame.display.flip()

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if the mouse click is within the square area
                mouse_x, mouse_y = event.pos
                if width * scale +10 < mouse_x < width * scale+10+square_size and 0 < mouse_y < square_size*len(color_matrix):
                    # remove select old rect
                    pygame.draw.rect(screen, selected_color, (width * scale + 10, selected_label * square_size, square_size, square_size))
                    # select new rect
                    selected_label = mouse_y // square_size
                    selected_color = color_matrix[selected_label]
                    print("Selected Color:", selected_color)
                # if select the image
                elif 0< mouse_x < width*scale and 0 < mouse_y < height*scale:
                    print("location on image:", mouse_x, mouse_y)
                    # update image
                    select_image_width  = mouse_x//scale
                    select_image_height = mouse_y//scale
                    colored_image[select_image_width, select_image_height] = selected_color
                    label_image[select_image_width, select_image_height] = selected_label
                    # redraw image
                    scaled_colored_image = np.repeat(np.repeat(colored_image, scale, axis=0), scale, axis=1)
                    surf = pygame.surfarray.make_surface(scaled_colored_image)
                    screen.blit(surf, (0, 0))
                    

        # Draw red empty rectangle around the selected square
        if selected_color is not None:
            pygame.draw.rect(screen, (255, 0, 0), (width * scale + 10, selected_label * square_size,
                                                    square_size, square_size), 2)

        # Update display
        pygame.display.flip()
    pygame.quit()
    
    # return image
    label_image = np.flipud(np.rot90(label_image, 1))
    return color_matrix, label_image