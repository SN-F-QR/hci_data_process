import os

import numpy
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def read_log(path, name, key_word, key_pos, key_len):
    # can read "point" for the unity log
    data = 0
    with open(os.path.join(path, name), 'r') as s_txt:
        for lines in s_txt:
            words = lines.split("_")
            if len(words) == key_len and words[key_pos] == key_word:
                temp = int(words[key_pos + 1])
                if data < temp:
                    data = temp

    return data


def walk_dir(path):
    # return all file names
    file_list = []
    temp = os.walk(path)
    for path, dirs, files in temp:
        if dirs:
            file_list = files

    return file_list


def read_convert(path):
    file = walk_dir(path)
    point = [[], [], []]
    number = []
    for name in file:
        name_s = name.split("_")
        if name_s[0] == "00":
            continue
        exp_index = int(name_s[1].replace(".txt", ""))  # judge index
        index = int(name_s[0])
        if len(number) > 0 and number[-1] != index:  # save people number
            number.append(index)
        elif len(number) == 0:
            number.append(index)

        point[exp_index].append(read_log(path, name, "Point", 1, 3))


def one_way_anova(data_group):
    # Shapiro–Wilk test
    print("Wilk: The null hypothesis cannot be rejected when p>0.05:")
    for tdata in data_group:
        print(stats.shapiro(tdata))
    print("ANOVA: The null hypothesis is no difference, and cannot be rejected when p>0.01/5")
    print(
        "F{free1},{free2} and p is:".format(free1=len(data_group) - 1, free2=len(data_group[0]) * len(data_group) + 1))
    print(stats.f_oneway(data_group[0], data_group[1], data_group[2]))


def fried_man_test(data_group):
    print("Friedman: The null hypothesis cannot be rejected when p>0.05:")
    print(stats.friedmanchisquare(data_group[0], data_group[1], data_group[2]))



def combine_qcsv(path):
    name = [r"csv\Survey for passthrough.csv", r"csv\Survey for mapping feature feeling.csv",
            r"csv\Survey for robot grasping feature feeling.csv"]

    data = []
    i = 1
    for k in name:
        df = pd.read_csv(os.path.join(path, k))
        group = [i] * df.shape[0]
        df = df.merge(pd.DataFrame({"group": group}), left_index=True, right_index=True)  # add group label
        i += 1
        df = df.rename(columns={'This feature is effective to finish the real-world task.': "effective",
                                'To what extent did this feature decrease the immersion on game?': "decrease",
                                'You could immerse yourself again quickly.': "again",
                                'You like this feature.': "like"})
        df = df.reindex(["group", "effective", "decrease", "again", "like"], axis="columns")
        data.append(df)
    return pd.concat(data, ignore_index=True)


if __name__ == "__main__":
    m_path = r"C:\Users\SN-F-\Developer\exp_data"
    m_name = r"csv\common_ques.csv"
    data = pd.read_csv(os.path.join(m_path, m_name))
    # save = combine_qcsv(m_path)
    judge = data.loc[:, "group"]
    tested = "decrease"
    m_group = [data.loc[judge == 1, tested], data.loc[judge == 2, tested], data.loc[judge == 3, tested]]
    # one_way_anova()
    fried_man_test(m_group)

    # print(stats.bartlett(point[0], point[1], point[2]))

    plt.figure(figsize=(5, 3), dpi=120, facecolor="white", edgecolor="red")
    plt.boxplot(m_group, labels=["Pass", "Map", "Robot"], showfliers=True, showmeans=True)
    plt.show()

    # csv_frame = pd.DataFrame({"number": number, "point_0": point[0], "point_1": point[1], "point_2": point[2]})
    # csv_frame.to_csv(os.path.join(m_path, "csv\sum.csv"), index=False)
