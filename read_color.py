import pygame
import pandas as pd

# Initialize pygame
pygame.init()

# Read the Excel file
df = pd.read_excel("Data\ColorStud\AliexpressColor.xlsx")

# Define screen dimensions and colors
screen_width = 550
screen_height = 550
rect_width = 87.5  # increased by 25%
rect_height = 62.5  # increased by 25%
margin = 10
black = (0, 0, 0)
white = (255, 255, 255)
font = pygame.font.SysFont(None, 15)

# Create screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Color Matrix")

# Function to determine if a color is dark or light
def is_dark(color):
    # Calculate the perceived brightness using the formula
    # 0.299*R + 0.587*G + 0.114*B
    brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return brightness < 128  # If brightness is less than 128, color is considered dark

# Function to draw rectangle with color
def draw_rect(x, y, color, text):
    pygame.draw.rect(screen, color, (x, y, rect_width, rect_height))
    if is_dark(color):
        text_color = white
    else:
        text_color = black
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + rect_width / 2, y + rect_height / 2))
    screen.blit(text_surface, text_rect)

# Main loop
running = True
while running:
    screen.fill(black)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw rectangles with colors from the DataFrame
    for i in range(6):
        for j in range(7):
            color_index = i * 6 + j
            if color_index < len(df):
                color = tuple(int(df["RGB"][color_index][1:][k:k+2], 16) for k in range(0, 6, 2))
                draw_rect(j * (rect_width + margin), i * (rect_height + margin), color, df["Name"][color_index])

    pygame.display.flip()

# Quit pygame
pygame.quit()
