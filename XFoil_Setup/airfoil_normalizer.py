import pandas as pd

if __name__ == '__main__':
    airfoil_filename = 'clark_y_coordinates'
    airfoil_filename_path = 'xfoil/' + airfoil_filename + '.txt'
    read_from_line = 2
    chord = 100.0063

    df = pd.read_csv(airfoil_filename_path, delim_whitespace=True, skiprows=(read_from_line-1), header=None)
    df.columns = ['x [mm]', 'y [mm]']

    normalized_df = df / chord

    normalized_df.to_csv(f'xfoil/{airfoil_filename}_normalized.txt', sep=' ', index=False)

    print(df)
    print(normalized_df)