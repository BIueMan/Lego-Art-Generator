import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tabulate import tabulate

# Read the file into a Pandas DataFrame
color_table = pd.read_excel('Data/ColorTable.xlsx')
stud_color = pd.read_excel('Data/StudColor.xlsx')

color_table = pd.DataFrame(color_table)
stud_color = pd.DataFrame(stud_color)

def get_set_parts(color):
    try:
        return stud_color[stud_color['Color'] == color]['Set Parts'].tolist()
    except:
        return ""

def get_element(color):
    try:
        return stud_color[stud_color['Color'] == color]['Element'].tolist()
    except:
        return ""

color_table['Set_Parts'] = color_table['Color'].apply(get_set_parts)
color_table['Element'] = color_table['Color'].apply(get_element)

color_table = color_table[color_table['RGB'].str.len() == 6]
color_table = color_table[color_table['Set_Parts'].str.len() != 0]

# Display the DataFrame
print(color_table.columns)

def hex_to_rgb(hex_color):
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b

def plot_rgb_points(rgb_colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color in rgb_colors:
        r, g, b = hex_to_rgb(color)
        ax.scatter(r, g, b, color=(r, g, b), marker='o')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    plt.show()
    # plt.waitforbuttonpress()

# Example usage:
plot_rgb_points(color_table['RGB'].astype(str))

table_str = tabulate(color_table, tablefmt='grid', headers='keys', showindex=False)
print(table_str)

lego_batman = pd.read_csv('Data/LegoArtSets/rebrickable_parts_31205-1-jim-lee-batman-collection.csv')

lego_batman = pd.DataFrame(lego_batman)
table_str = tabulate(lego_batman, tablefmt='grid', headers='keys', showindex=False)
print(table_str)