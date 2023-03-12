import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def read_convert(path, name, key_word, key_pos, key_len):
    # can read point
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
        file_list = files

    return file_list


if __name__ == "__main__":
    m_path = r"C:\Users\SN-F-\Developer\exp_data"
    m_file = walk_dir(m_path)
    point = [[], [], []]
    number = []
    for m_name in m_file:
        name_s = m_name.split("_")
        if name_s[0] == "00" or name_s[0] == "sum.csv":
            continue
        exp_index = int(name_s[1].replace(".txt", ""))  # judge index
        index = int(name_s[0])
        if len(number) > 0 and number[-1] != index:  # save people number
            number.append(index)
        elif len(number) == 0:
            number.append(index)

        point[exp_index].append(read_convert(m_path, m_name, "Point", 1, 3))

    for a_point in point:
        print(stats.shapiro(a_point))

    print(stats.f_oneway(point[0], point[1], point[2]))

    plt.figure(figsize= (5,3), dpi=120, facecolor="white", edgecolor="red")
    plt.boxplot(point, labels=["Pass", "Map", "Robot"])
    plt.show()



    # csv_frame = pd.DataFrame({"number": number, "point_0": point[0], "point_1": point[1], "point_2": point[2]})
    # csv_frame.to_csv(os.path.join(m_path, "sum.csv"), index=False)



