import pandas as pd
import argparse
from os import path

write_directory = 'data/'


def get_data(file_path):
    """function that takes a file on the web and saves it locally"""
    df = pd.read_csv(file_path, sep=',')
    return df.to_csv(path.join(write_directory) + file_path.split('/')[-1])


if __name__ == '__main__':

    # instantiate ArgumentParser class
    parser = argparse.ArgumentParser()

    # add argument to parser object
    parser.add_argument("file_path", type=str, help='enter file you want to download')

    # the argument passed into the parser is
    args = parser.parse_args()

    # pass this is argument to the file_path parameter
    get_data(file_path=args.file_path)








