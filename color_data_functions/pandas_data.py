import pandas as pd
import matplotlib.pyplot as plt
import os
from tabulate import tabulate
import numpy as np

# Read the file into a Pandas DataFrame
def get_color_table(color_table_path:str = 'Data/ColorStud/ColorTable.xlsx',
                    stud_color_path:str  = 'Data/ColorStud/StudColor.xlsx',
                    round_color_path:str = 'Data/ColorStud/RoundColor.xlsx'):
    ## read tables
    color_table = pd.read_excel(color_table_path)   # main table
    stud_color = pd.read_excel(stud_color_path)     # sub table
    round_color = pd.read_excel(round_color_path)   # sub table
    
    color_table = pd.DataFrame(color_table)
    stud_color = pd.DataFrame(stud_color)
    round_color = pd.DataFrame(round_color)

    ## copy data from sub table to main table
    def get_element(color, *args):
        pandas = args[0]
        element = args[1]
        try:
            return pandas[pandas['Color'] == color][element].tolist()
        except:
            return ""
    color_table['SetParts_Stud']  = color_table['Color'].apply(lambda x: get_element(x, stud_color, 'Set Parts'))
    color_table['Element_Stud']   = color_table['Color'].apply(lambda x: get_element(x, stud_color, 'Element'))
    color_table['SetParts_Round'] = color_table['Color'].apply(lambda x: get_element(x, round_color, 'Set Parts'))
    color_table['Element_Round']  = color_table['Color'].apply(lambda x: get_element(x, round_color, 'Element'))
    
    ## remove unwanted rows
    color_table = color_table[color_table['RGB'].str.len() == 6]                                                            # if RGB not present
    color_table = color_table[list(~np.isnan(color_table['SetParts_Stud'].str[0]) | ~np.isnan(color_table['SetParts_Round'].str[0]))]  # if part not exists
    
    return color_table

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

def get_lego_art_set_dict(path_to_dir:str = 'Data/LegoArtSets'):
    set_dict = {}
    for file in os.listdir(path_to_dir):
        set_name = path_to_dir.split('-1-')[-1].split('.csv')[0]
        lego_art_set = pd.read_csv(f'{path_to_dir}/{file}')
        lego_art_set = pd.DataFrame(lego_art_set)
        set_dict[set_name] = lego_art_set

if __name__ == "__main__":
    # Display the DataFrame
    color_table = get_color_table()
    table_str = tabulate(color_table, tablefmt='grid', headers='keys', showindex=False)
    print(table_str)
    # plot table color in 3D-RGB
    plot_rgb_points(color_table['RGB'].astype(str))
