import matplotlib.pyplot as plt
import pandas as pd
import re

if __name__ == '__main__':
    name = 'clark_y_polars_w_20'
    polar_filename = 'xfoil/' + name + '.txt'
    airfoil_chord = 100.00636
    single_plot = False

    with open(polar_filename, 'r') as file:
        data = file.read()

    # add a space before asterisks following digits
    modified_data = re.sub(r'(?<=\d)\*+', ' *', data)

    # write the content back to file
    with open(polar_filename, 'w') as file:
        file.write(modified_data)

    # read in polar file from x-foil:
    df = pd.read_csv(polar_filename, delim_whitespace=True, skiprows=12, header=None)

    df.columns = ['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr_Relative', 'Bot_Xtr_Relative']
    
    df.to_csv(f'{name}.csv', index=False)

    # plot

    plt.rcParams['font.size'] = 16.

    if single_plot == True:
        plt.plot(df['alpha'], df['CL'], marker='.', linestyle='-')
        plt.xlabel('alpha (deg)')
        plt.ylabel('CL')
        plt.title('Polar Plot for CL')
        plt.show()
    else:
        num_plots = 4 #len(df.columns)
        num_columns = 4
        num_rows = (num_plots + num_columns - 1) // num_columns

        # create subplots
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 5 * num_rows))
        axs = axs.flatten()

        # plot CL vs CD
        axs[0].plot(df['CD'], df['CL'], marker='.')
        axs[0].set_title('CL vs CD')
        axs[0].set_xlabel('CD')
        axs[0].set_ylabel('CL')
        
        # plot CL vs CD
        axs[1].plot(df['alpha'], df['CL'], marker='.')
        axs[1].set_title('CL vs alpha')
        axs[1].set_xlabel('alpha (degrees)')
        axs[1].set_ylabel('CL')

       # plot CL vs CD
        axs[2].plot(df['alpha'], df['CD'], marker='.')
        axs[2].set_title('CD vs alpha')
        axs[2].set_xlabel('alpha (degrees)')
        axs[2].set_ylabel('CD')

        # plot CL vs CD
        axs[3].plot(df['alpha'], df['CM'], marker='.')
        axs[3].set_title('CM vs alpha')
        axs[3].set_xlabel('alpha (degrees)')
        axs[3].set_ylabel('CM')

        # # plot all columns vs alpha
        # for i, column in enumerate(df.columns[1:], start=1):
        #     axs[i].plot(df['alpha'], df[column], marker='.')
        #     axs[i].set_title(f'{column} vs alpha')
        #     axs[i].set_xlabel('alpha (deg)')
        #     axs[i].set_ylabel(column)

        for j in range(num_plots, num_rows * num_columns):
            fig.delaxes(axs[j])

        #fig.suptitle(f"{name}", fontsize=16)
        plt.tight_layout()
        plt.show()