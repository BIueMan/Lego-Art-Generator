import pandas as pd

def read_color_list():
    # Read the text file
    with open('Data\color_list.txt', 'r') as file:
        data = file.readlines()

    # Initialize lists to store parsed data
    names = []
    colors = []
    numbers = []

    # Parse the data
    for line in data:
        if line.startswith('#'):
            continue
        parts = line.strip().split(', ')
        for part in parts:
            key, value = part.split(' - ')
            if key == 'name':
                name_parts = value.split(' ')
                number = name_parts[-1]
                name = ' '.join(name_parts[:-1])
                names.append(name)
                numbers.append(number)
            elif key == 'color':
                colors.append(value)

    # Create a DataFrame
    df = pd.DataFrame({'Name': names, 'Color': colors, 'Number': numbers})
    return df

if __name__ == "__main__":
    df = read_color_list()
    print(df)
