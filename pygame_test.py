import pygame
import sys
import os

def create_screen(image_rect):
    # Set up display
    window_width = image_rect.width + 50
    window_height = image_rect.height
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
        buttons.append({'rect':button_rect, 'font':font.render(str(i), True, (0, 0, 0)), 'color': (0,255,0)})
    return buttons

def get_pixel_color(image, pos):
    # Get the color of the pixel at the specified position
    color = image.get_at(pos)
    return color

def run_app(image, image_rect, screen, buttons):
    # List to store selected points

    running = True
    selected_butten = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Left mouse button clicked
                mouse_pos = event.pos

                # Check if any button is clicked
                for num, button in enumerate(buttons):
                    if button['rect'].collidepoint(mouse_pos):
                        # print("Button Clicked:", num)
                        selected_butten = num

                # Check if clicked on the image area
                if image_rect.collidepoint(mouse_pos):
                    # Add point to the list
                    buttons[selected_butten]['location'] = mouse_pos
                    buttons[selected_butten]['color'] = get_pixel_color(image, mouse_pos)[:3]
                    print(f'num: {selected_butten} = {mouse_pos}, color - {buttons[selected_butten]["color"]}')

        screen.blit(image, image_rect)

        # Draw red dots at selected points
        for num, button in enumerate(buttons):
            if 'location' in buttons[num]:
                pygame.draw.circle(screen, (255, 0, 0), buttons[num]['location'], 5)

        # Draw buttons
        for num, button in enumerate(buttons):
            pygame.draw.rect(screen, button['color'], button['rect'])
            if selected_butten == num:
                pygame.draw.rect(screen, (255, 0, 0), button['rect'], 3)
            screen.blit(button['font'], button['rect'].center)


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
