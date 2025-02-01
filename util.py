import numpy as np
from numpy.linalg import norm
import csv

def load_data(directory, parameter):
    values = []
    # Load people
    with open(directory, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if (parameter == "Intensity"):
            for row in reader:
                if (-4000<float(row["Wavenumber"])<=0):
                    values.append(row[parameter])
        else:
            for row in reader:
                values.append(row[parameter])
    return values

def stringtofloat(array):
        float_array = [float(string) for string in array]
        return float_array

def cosine_similarity(A, B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine