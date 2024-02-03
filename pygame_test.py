import pygame
import sys
import os

def create_screen(image_rect):
    # Set up display
    window_width = image_rect.width + 200
    window_height = max(image_rect.height, 600)
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Point Selector")
    return screen

def create_buttons(image_rect):
    # Create font for button labels
    font = pygame.font.Font(None, 36)

    # Create buttons
    buttons = []
    button_height = 40
    for i in range(7):
        button_rect = pygame.Rect(image_rect.width + 10, 50 + i * (button_height + 10), 50, button_height)
        buttons.append((button_rect, i, font.render(str(i), True, (0, 0, 0))))
    return buttons

def run_app(image, image_rect, screen, buttons):
    # List to store selected points
    selected_points = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left mouse button clicked
                mouse_pos = event.pos

                # Check if any button is clicked
                for button_rect, button_number, _ in buttons:
                    if button_rect.collidepoint(mouse_pos):
                        print("Button Clicked:", button_number)

                # Check if clicked on the image area
                if image_rect.collidepoint(mouse_pos):
                    # Add point to the list
                    selected_points.append(mouse_pos)
                    print("Selected Points:", selected_points)

        screen.blit(image, image_rect)

        # Draw red dots at selected points
        for point in selected_points:
            pygame.draw.circle(screen, (255, 0, 0), point, 5)

        # Draw buttons
        for button_rect, _, button_text in buttons:
            pygame.draw.rect(screen, (0, 255, 0), button_rect)
            screen.blit(button_text, button_rect.center)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

def main():
    pygame.init()

    # Input image
    image_path = os.path.join("Data", "Lenna.png")
    image = pygame.image.load(image_path)
    image_rect = image.get_rect()

    # Create screen
    screen = create_screen(image_rect)

    # Create buttons
    buttons = create_buttons(image_rect)

    # Run the app
    run_app(image, image_rect, screen, buttons)

if __name__ == "__main__":
    main()
