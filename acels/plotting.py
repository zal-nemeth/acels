# import pandas as pd
# import matplotlib.pyplot as plt

# # Example for one table
# df = pd.read_csv('acels/analysis/extended_50_mae_non_quant_models.csv')

# fig, ax = plt.subplots()
# ax.axis('tight')
# ax.axis('off')
# ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# plt.savefig('table.svg', format='svg')


import pandas as pd
import matplotlib.pyplot as plt

def create_svg_from_csv(csv_file, title):
    # Read the CSV file
    df = pd.read_csv(csv_file, index_col=0)  # Adjust if your index is differently located

    # Creating the plot
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values.round(5), colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')

    plt.title(title)

    # Save the figure
    svg_file = csv_file.replace('.csv', '.svg')
    plt.savefig(svg_file, format='svg')
    print(f'Saved SVG file: {svg_file}')

# Example usage
csv_file = 'acels/analysis/extended_50_mae_non_quant_models.csv'  # Update this path
title = 'Your Title Here'  # Update this title
create_svg_from_csv(csv_file, title)
