
import numpy as np
from matplotlib import pyplot as plt


def get_data():
    data = []
    filename = "developer_hiring_data.csv"
    with open(filename) as file:
        #skip the first line
        file.readline()
        for line in file.readlines():
            origin, age, gender, education, degree, language, job_level, current_role, hired = line.strip().split(',')
            age = int(age)
            data.append([origin, age, gender, education, degree, language, job_level, current_role, hired])
    return data[:5]