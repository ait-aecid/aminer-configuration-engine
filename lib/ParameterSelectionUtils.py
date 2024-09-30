import pandas as pd
import numpy as np
import itertools
import os
import yaml
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import math
import networkx as nx

def copy_and_save_file(input_file, output_file, line_numbers):
    """Write specified lines of the input file into the output file."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        for line_number in line_numbers:
            if 0 <= line_number < len(lines):
                outfile.write(lines[line_number])

def concatenate_files(input_files, output_file):
    """Concatenate multiple files into one file."""
    with open(output_file, 'w') as outfile:
        for input_file in input_files:
            with open(input_file, 'r') as infile:
                outfile.write(infile.read())
