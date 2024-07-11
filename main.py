import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from tool import Toolbox
# import scikit_posthocs as sp




def time_f(x):
    # convert string time to float second
    return float(x.split(":")[0])*60 + float(x.split(":")[1])


def read_log(path, name, key_word, key_pos, key_len):
    # can read "point" for the unity log
    real_flag = False
    data = 0
    open_t = []
    close_t = []
    file_path = os.path.join(path, name)
    with open(file_path, 'r') as s_txt:
        for lines in s_txt:
            words = lines.split("_")
            # avoid fake switch
            if key_word.split(" ")[-1] == "Set":
                if len(words) == key_len and words[key_pos] == "Ask for Action":
                    real_flag = True

            if len(words) == key_len and words[key_pos] == key_word:
                if key_word == "Point":
                    data = int(words[key_pos + 1])
                elif key_word == "Shot":
                    data += 1
                elif key_word.split(" ")[-1] == "Set" and real_flag:
                    # open feature
                    if words[key_pos + 1].replace("\n", "") == "True":
                        open_t.append(time_f(words[0]))
                    elif len(open_t) > len(close_t):
                        close_t.append(time_f(words[0]))
                        real_flag = False

    if key_word.split(" ")[-1] == "Set":
        data = find_around_action(file_path, open_t, close_t, bias=5)

    return data


def find_around_action(path, open_t, close_t, bias):
    # find out the data before and after some actions
    assert len(open_t) == len(close_t) == 3
    for i in range(1,len(open_t)):
        assert open_t[i]-bias > close_t[i-1]+bias
    shot_before = []
    hit_before = []
    shot_after = []
    hit_after = []
    sb,sa,hb,ha = 0,0,0,0  # can be reduced to s,h

    i = 0
    # find the data before and after action
    with open(path, 'r') as s_txt:
        for lines in s_txt:
            words = lines.split("_")
            if len(words) <= 1:  # no time log in this line
                continue
            cur_time = time_f(words[0])
            if open_t[i]-bias <= cur_time <= open_t[i]:
                if words[1] == "Shot":
                    sb += 1
                elif words[1] == "Hit":
                    hb += 1
            elif cur_time > open_t[i] and len(shot_before)!=i+1:
                shot_before.append(sb)
                hit_before.append(hb)
                sb = 0
                hb = 0
            if close_t[i] <= cur_time <= close_t[i] + bias:
                if words[1] == "Shot":
                    sa += 1
                elif words[1] == "Hit":
                    ha += 1
            elif cur_time > close_t[i] + bias and len(shot_after)!=i+1:
                shot_after.append(sa)
                hit_after.append(ha)
                sa = 0
                ha = 0
                i += 1
                if i == 3:
                    break

    assert len(shot_before) == len(shot_after) == 3
    accuracy = [0.0, 0.0]
    med_b = []
    med_a = []
    for i in range(len(shot_before)):
        if shot_before[i] == 0 or shot_after[i] == 0:
            continue
        med_b.append(hit_before[i] / shot_before[i])
        med_a.append(hit_after[i] / shot_after[i])
    accuracy[0] = np.median(med_b)
    accuracy[1] = np.median(med_a)
    # accuracy[0] = sum(hit_before) / sum(shot_before)
    # accuracy[1] = sum(hit_after) / sum(shot_after)
    # accuracy[0] = hit_before[1] / shot_before[1]
    # accuracy[1] = hit_after[1] / shot_after[1]

    return accuracy[1]-accuracy[0]


def read_convert(path):
    # read all experiment files, convert to csv
    files = Toolbox.walk_dir(path)
    group_action = ["Camera Set", "Reflection Set", "Robot Menu Set"]
    data_dict = {"sub_id": [],
                 "group": [],
                 "point": [],
                 "accuracy": [],
                 "acc_difference": []}
    for name in files:
        name_s = name.split("_")
        if name_s[0] == "00":
            continue
        sub_id = int(name_s[0])
        group = int(name_s[1].replace(".txt", ""))
        data_dict["sub_id"].append(sub_id)
        data_dict["group"].append(group)

        point = read_log(path, name, "Point", 1, 3)
        data_dict["point"].append(point)  # count point
        data_dict["accuracy"].append(point/read_log(path, name, "Shot", 1, 5))  # count shot times
        data_dict["acc_difference"].append(read_log(path, name, group_action[group], 1, 3))

    # convert to dataframe
    df = pd.DataFrame(data_dict)
    # analyze data
    # TODO: generalize this function?
    # analyze_nasa(df, 3, 2)
    return df


def one_way_anova(data_group):
    # Shapiro–Wilk test
    print("Wilk: The null hypothesis cannot be rejected when p>0.05:")
    for tdata in data_group:
        print(stats.shapiro(tdata))
    print("ANOVA: The null hypothesis is no difference, and cannot be rejected when p>0.01/5")
    print(
        "F{free1},{free2} and p is:".format(free1=len(data_group) - 1, free2=len(data_group[0]) * len(data_group) + 1))
    print(stats.f_oneway(data_group[0], data_group[1], data_group[2]))


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




def delete_outlier(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    low, up = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier = s.mask((s < low) | (s > up))
    return outlier

def analyze_questionnaire(path):
    name = r"csv\common_ques.csv"
    data = pd.read_csv(os.path.join(path, name))
    judge = data.loc[:, "group"]
    tested = "decrease"
    group = [data.loc[judge == 1, tested], data.loc[judge == 2, tested], data.loc[judge == 3, tested]]
    Toolbox.fried_man_test(group)

    # print(stats.bartlett(point[0], point[1], point[2]))

    plt.figure(figsize=(5, 3), dpi=120, facecolor="white", edgecolor="red")
    plt.boxplot(group, labels=["Pass", "Map", "Robot"], showfliers=True, showmeans=True)
    plt.show()

if __name__ == "__main__":
    m_path_NASA = os.path.expanduser("~/Developer/Exp_Result/NASA_TLX")
    m_path_exp = r"C:\Users\SN-F-\Developer\exp_data\Exp_game"
    m_path = r"C:\Users\SN-F-\Developer\exp_data"

    nasa_handler = NasaTLXProcess(m_path_NASA, ["NoRS", "Arr.", "High.", "Swap"])
    nasa_handler.read_nasa()
    group_lists = nasa_handler.rs_data['group'].tolist()
    for index, group_name in enumerate(group_lists):
        if group_name == 2:
            group_lists[index] = 3
        elif group_name == 3:
            group_lists[index] = 2
    nasa_handler.rs_data['group'] = pd.Series(group_lists)
    nasa_handler.analyze_nasa(nasa_handler.rs_data, nasa_handler.group_names, start=2, plot=True)
    # read_convert(m_path_exp)
    # print(1)
    # analyze_questionnaire(m_path)



    # csv_frame = pd.DataFrame({"number": number, "point_0": point[0], "point_1": point[1], "point_2": point[2]})
    # csv_frame.to_csv(os.path.join(m_path, "csv\sum.csv"), index=False)
