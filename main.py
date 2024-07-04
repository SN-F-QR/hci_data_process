import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
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


def walk_dir(path):
    # return all file names
    file_list = []
    temp = os.walk(path)
    for path, dirs, files in temp:
        file_list = files

    return file_list


def read_convert(path):
    # read all experiment files, convert to csv
    files = walk_dir(path)
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
    analyze_nasa(df, 3, 2)
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


def fried_man_test(data_group):
    # TODO: add test mode support, find suitable post-hoc test
    print("Friedman: The null hypothesis cannot be rejected when p>0.05:")
    p = stats.friedmanchisquare(*data_group)
    # p = stats.kruskal(data_group[0], data_group[1], data_group[2])
    # p = stats.median_test(data_group[0], data_group[1], data_group[2], nan_policy='omit')
    print(p)
    if p[1] < 0.05:
        print("Found significant difference, run wilcoxon post-hoc test")
        print("Wilcoxon: Reject the null hypothesis that there is no difference when p<0.05")
        print(stats.wilcoxon(data_group[0], data_group[1], correction=True, method="approx", alternative="two-sided", nan_policy="omit"))
        print(stats.wilcoxon(data_group[0], data_group[2], correction=True, method="approx", alternative="two-sided", nan_policy="omit"))
        print(stats.wilcoxon(data_group[1], data_group[2], correction=True, method="approx", alternative="two-sided", nan_policy="omit"))


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


def read_nasa_raw(path, file_name, raw_type):
    assert raw_type == 'pw' or raw_type == 'rs'
    # dict for dataframe
    data_dict = {"sub_id": [],
                 "group": [],
                 "mental": [],
                 "physical": [],
                 "temporal": [],
                 "performance": [],
                 "effort": [],
                 "frustration": []}

    for name in file_name:
        n_path = os.path.join(path, name)
        with open(n_path, 'r') as csv_txt:
            data_flag = False
            weight = [0 for i in range(6)]  # store weight for pw and score for raw
            for line in csv_txt:
                words = line.split(",")
                if words[0] == "SUBJECT ID:":
                    data_dict["sub_id"].append(int(words[1]))
                    continue
                if words[0] == "STUDY GROUP:":
                    data_dict["group"].append(int(words[1]))
                    continue
                if words[0] == "PAIRWISE CHOICES" or words[0] == "RATING SCALE:":
                    data_flag = True
                    continue
                if data_flag and len(words) > 1:
                    if raw_type == 'pw':
                        index = count_weight(words[-1].replace("\n", ""))
                        weight[index] += 1
                    elif raw_type == 'rs':
                        index = count_weight(words[0])
                        weight[index] = int(words[-1])

            data_dict["mental"].append(weight[0])
            data_dict["physical"].append(weight[1])
            data_dict["temporal"].append(weight[2])
            data_dict["performance"].append(weight[3])
            data_dict["effort"].append(weight[4])
            data_dict["frustration"].append(weight[5])

    raw_frame = pd.DataFrame(data_dict)
    return raw_frame


def count_weight(tag):
    if tag == "Mental Demand":
        return 0
    elif tag == "Physical Demand":
        return 1
    elif tag == "Temporal Demand":
        return 2
    elif tag == "Performance":
        return 3
    elif tag == "Effort":
        return 4
    elif tag == "Frustration":
        return 5


def analyze_nasa(data_frame, group_names, start, plot=True):
    group_judge = data_frame.loc[:, "group"]  # get group number
    group_count = data_frame["group"].nunique()
    assert group_count == len(group_names)
    plt_count = 0
    if plot:
        fig, axes = plt.subplots(2, 3, layout='constrained')

    for i in range(start, data_frame.shape[1]):
        group_data = []
        group_data_plt = []
        for j in range(group_count):
            # TODO: Mind here, deal with nan or not
            group_data.append(delete_outlier(data_frame.iloc[(group_judge == j).values, i]))
            group_data_plt.append((data_frame.iloc[(group_judge == j).values, i]))

        if plot:
            axes.flat[plt_count].boxplot(group_data_plt, labels=group_names, showfliers=True, showmeans=True)
            axes.flat[plt_count].set_title(data_frame.columns[i])
            plt_count += 1
        print("------" + data_frame.columns[i] + " result:")
        # fried_man_test(group_data_plt)
        # one_way_anova(group_data_plt)

    if plot:
        plt.show()


def delete_outlier(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    low, up = q1 - 1.5*iqr, q3 + 1.5*iqr
    outlier = s.mask((s < low) | (s > up))
    return outlier


def read_nasa(path, group_names):
    # TODO: Add first_time support, verify weighted result
    # only handle NASA_TLX with or without each-time pairwise
    # *be careful the file order*
    # get file lists
    file = walk_dir(path)
    pw_file = []
    rs_file = []
    for name in file:
        names = name.split("_")
        if names[4] == "PW":
            pw_file.append(name)
        elif names[4] == "RS":
            rs_file.append(name)

    weighted = len(pw_file) == len(rs_file)
    if len(pw_file) > 0:
        assert weighted

    # read scale result
    rs_frame = read_nasa_raw(path, rs_file, 'rs')
    # get raw result
    print("---NASA TLX raw result---")
    analyze_nasa(rs_frame, group_names, 2, plot=True)
    # get weighted result
    if weighted:
        # read pair result and get weight
        pw_frame = read_nasa_raw(path, pw_file, 'pw')
        print("---NASA TLX weighted result---")
        weighted_frame = rs_frame
        weighted_frame.loc[:, "mental":] = rs_frame.loc[:, "mental":] * pw_frame.loc[:, "mental":] / 15
        analyze_nasa(weighted_frame, group_names, 2, plot=True)
    # weighted_frame.loc[:, "sum"] = weighted_frame.iloc[:, 2]
    # for i in range(3,8):
        # weighted_frame.loc[:, "sum"] += weighted_frame.iloc[:, i]
    # return 6*2 results


def analyze_questionnaire(path):
    name = r"csv\common_ques.csv"
    data = pd.read_csv(os.path.join(path, name))
    judge = data.loc[:, "group"]
    tested = "decrease"
    group = [data.loc[judge == 1, tested], data.loc[judge == 2, tested], data.loc[judge == 3, tested]]
    fried_man_test(group)

    # print(stats.bartlett(point[0], point[1], point[2]))

    plt.figure(figsize=(5, 3), dpi=120, facecolor="white", edgecolor="red")
    plt.boxplot(group, labels=["Pass", "Map", "Robot"], showfliers=True, showmeans=True)
    plt.show()

if __name__ == "__main__":
    m_path_NASA = os.path.expanduser("~/Developer/Exp_Result/NASA_TLX")
    m_path_exp = r"C:\Users\SN-F-\Developer\exp_data\Exp_game"
    m_path = r"C:\Users\SN-F-\Developer\exp_data"
    read_nasa(m_path_NASA, group_names=["NoRS", "Arr.", "Swap", "High."])
    # read_convert(m_path_exp)
    # print(1)
    # analyze_questionnaire(m_path)



    # csv_frame = pd.DataFrame({"number": number, "point_0": point[0], "point_1": point[1], "point_2": point[2]})
    # csv_frame.to_csv(os.path.join(m_path, "csv\sum.csv"), index=False)
