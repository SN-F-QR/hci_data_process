import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool import Toolbox

class DataProcess:
    def __init__(self, path, group_names):
        self.path = path  # path to the document includes data file(s)
        self.group_names = group_names  # experimental groups
        self.group_num = len(group_names)
        self.file_names = Toolbox.walk_dir(self.path)  # record all data files
        self.df = pd.DataFrame()

        self.plots = []  # record all plots in matplot for customization

    # Assume that file name is like: expName_groupNAME_participantNo._xxx
    # def read_data(self, where_group=1, where_user=2):

    """
    Using subplots to plot the dataframe
        Start: the first column index of data
        fig_design: (2,3) => 2 x 3 subplot design
        group_colors: colors for each group
    """
    # TODO: Add fig_design 1x6 and 1x1?
    def plot_sub_data(self, start=2, fig_design=(2, 3), subplot_titles=None, group_colors=None, flier_mark='o'):
        if group_colors is None:
            group_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']
        group_judge = self.df.loc[:, "group"]  # get group number for further filtering
        assert self.df["group"].nunique() == self.group_num
        # plt_count = 0
        max_sig_count = 0  # To decide the max height of each subplot
        fig, axes = plt.subplots(fig_design[0], fig_design[1], layout="constrained")
        self.plots = axes
        # fig, axes = plt.subplots(1, 6, layout="constrained", figsize=(12, 4)) # for two columns

        for i in range(start, self.df.shape[1]):  # column nums, namely the dependent variables
            group_data = []  # store all data in same group for comparision
            plt_index = i - start
            for j in range(self.group_num):
                # group_data.append(delete_outlier(data_frame.iloc[(group_judge == j).values, i]))
                group_data.append((self.df.iloc[(group_judge == j).values, i]))

            # Subplot Setting
            # supported color
            # bp_colors = ["#5184B2", "#AAD4F8", "#F1A7B5", "#D55276", "#F2F5FA"]
            if group_colors is None:
                group_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']
            bp = axes.flat[plt_index].boxplot(group_data, patch_artist=True, labels=self.group_names,
                                              showfliers=True, showmeans=True)

            for patch, color in zip(bp["boxes"], group_colors):
                patch.set_facecolor(color)
                patch.set_linewidth(0.5)
                # patch.set_edgecolor('white')
            for flier in bp["fliers"]:
                flier.set(marker=flier_mark)
            for mean in bp["means"]:
                mean.set_markerfacecolor("#86C166")
                mean.set_markeredgecolor("#86C166")

            if subplot_titles is None:
                axes.flat[plt_index].set_title(self.df.columns[i])
            else:
                axes.flat[plt_index].set_title(subplot_titles[plt_index])

            # Draw significant lines if any
            sig_group = self.significance_test(self.df.columns[i], group_data)
            if len(sig_group) > 0:
                max_sig_count = max(max_sig_count, len(sig_group))
                height_add = 0
                height_basic = axes.flat[plt_index].get_ylim()[1]
                axes.flat[plt_index].set_ylim(-5, height_basic + len(sig_group) * 15)
                axes.flat[plt_index].set_yticks(np.arange(0, 101, 20))
                for res in sig_group:
                    # Get x_axis positions for start and end
                    x_s = axes.flat[plt_index].get_xticks()[res[0]]
                    x_e = axes.flat[plt_index].get_xticks()[res[1]]
                    self.add_significance(x_s, x_e, height_basic + height_add, res[2], axes.flat[plt_index])
                    height_add += 15

        # Adjust height for all plot to maintain same y-axis
        for ax in axes.flat:
            ax.set_ylim(-5, 100 + max_sig_count * 15)
            ax.set_yticks(np.arange(0, 101, 20))
        # fig.tight_layout()
        # plt.savefig("NASA_TLX_new.pdf", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # Re-write this if other analyze method is expected
    def significance_test(self, name, data, correction=False):
        print("-------" + name + " result:")
        p_value = Toolbox.fried_man_test(data)[1]
        sig_group = []
        # include borderline condition
        if p_value < 0.06:
            sig_group = Toolbox.wilcoxon_post_hoc(data, bonferroni_holm=correction)

        return sig_group

    @staticmethod
    def add_significance(start, end, height, p_value, ax):
        x = [start, start, end, end]
        y = [height, height + 1, height + 1, height]
        ax.plot(x, y, color="k", linewidth=1.5)

        sign = ""
        if p_value < 0.001:
            sign = "***"
        elif p_value < 0.01:
            sign = "**"
        elif p_value <= 0.05:
            sign = "*"

        ax.text((start + end) / 2, height + 1, sign, ha="center", va="bottom")


class NasaTLXProcess(DataProcess):
    def __init__(self, path, group_names):
        super().__init__(path, group_names)
        self.rs_data = pd.DataFrame()
        self.pw_data = pd.DataFrame()

    # only handle NASA_TLX with or without each-time pairwise
    def read_nasa(self):
        pw_file = []
        rs_file = []
        for name in self.file_names:
            names = name.split("_")
            if names[4] == "PW":
                pw_file.append(name)
            elif names[4] == "RS":
                rs_file.append(name)

        # judge is raw TLX or paired TLX
        weighted = len(pw_file) == len(rs_file)
        if len(pw_file) > 0:
            assert weighted

        # read scale result
        self.rs_data = self.read_nasa_raw('rs')
        # get raw result
        print("---NASA TLX raw result---")
        # self.analyze_nasa(self.rs_data, self.group_names, start=2, plot=True)
        # get weighted result
        if weighted:
            # read pair result and get weight
            pw_frame = self.read_nasa_raw('pw')
            print("---NASA TLX weighted result---")
            self.pw_data = self.rs_data[:]  # avoid ref
            self.pw_data.loc[:, "mental":] = self.rs_data.loc[:, "mental":] * pw_frame.loc[:, "mental":] / 15

        # weighted_frame.loc[:, "sum"] = weighted_frame.iloc[:, 2]
        # for i in range(3,8):
        # weighted_frame.loc[:, "sum"] += weighted_frame.iloc[:, i]
        # return 6*2 results

    def read_nasa_raw(self, raw_type):
        assert raw_type == 'pw' or raw_type == 'rs'
        # dict for dataframe
        data_dict = {"sub_id": [],
                     "group": [],
                     "Mental Demand": [],
                     "Physical Demand": [],
                     "Temporal Demand": [],
                     "Performance": [],
                     "Effort": [],
                     "Frustration": []}

        for name in self.file_names:
            full_path = os.path.join(self.path, name)
            with open(full_path, 'r') as csv_txt:
                data_flag = False
                weight = {"Mental Demand": 0,
                          "Physical Demand": 0,
                          "Temporal Demand": 0,
                          "Performance": 0,
                          "Effort": 0,
                          "Frustration": 0}  # store weight for pw and score for raw
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
                            weight[words[-1].replace("\n", "")] += 1
                        elif raw_type == 'rs':
                            weight[words[0]] = int(words[-1])

                for key in weight.keys():
                    data_dict[key].append(weight[key])

        raw_frame = pd.DataFrame(data_dict)
        return raw_frame

    def analyze_nasa(self, data_frame, group_names, start=2, plot=True):
        group_judge = data_frame.loc[:, "group"]  # get group number
        group_count = data_frame["group"].nunique()
        assert group_count == len(group_names)
        plt_count = 0
        max_sig_count = 0
        if plot:
            # fig, axes = plt.subplots(1, 6, layout="constrained", figsize=(12, 4)) # for two columns
            fig, axes = plt.subplots(2, 3, layout="constrained")  # smaller space

        for i in range(start, data_frame.shape[1]):  # column nums, namely the dependent variables
            group_data = []
            group_data_plt = []
            for j in range(group_count):
                # TODO: Mind here, deal with nan or not
                group_data.append(delete_outlier(data_frame.iloc[(group_judge == j).values, i]))
                group_data_plt.append((data_frame.iloc[(group_judge == j).values, i]))

            if plot:
                # bp_colors = ["#5184B2", "#AAD4F8", "#F1A7B5", "#D55276", "#F2F5FA"]
                bp_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']
                bp = axes.flat[plt_count].boxplot(group_data_plt, patch_artist=True, labels=group_names,
                                                  showfliers=True, showmeans=True)

                for patch, color in zip(bp["boxes"], bp_colors):
                    patch.set_facecolor(color)
                    patch.set_linewidth(0.5)
                    # patch.set_edgecolor('white')
                for flier in bp["fliers"]:
                    flier.set(marker='*')
                for mean in bp["means"]:
                    mean.set_markerfacecolor("#86C166")
                    mean.set_markeredgecolor("#86C166")
                axes.flat[plt_count].set_title(data_frame.columns[i])
            print("------" + data_frame.columns[i] + " result:")
            # axes.flat[plt_count].set_ylim(-5, 100 + 3 * 15)
            # axes.flat[plt_count].set_yticks(np.arange(0, 101, 20))

            p_value = Toolbox.fried_man_test(group_data_plt)[1]
            if p_value < 0.06:
                sig_group = Toolbox.wilcoxon_post_hoc(group_data_plt)
                max_sig_count = max(max_sig_count, len(sig_group))
                height_add = 0
                height_basic = axes.flat[plt_count].get_ylim()[1]
                axes.flat[plt_count].set_ylim(-5, height_basic + len(sig_group) * 15)
                axes.flat[plt_count].set_yticks(np.arange(0, 101, 20))
                for res in sig_group:
                    # Get x_axis positions for start and end
                    x_s = axes.flat[plt_count].get_xticks()[res[0]]
                    x_e = axes.flat[plt_count].get_xticks()[res[1]]
                    NasaTLXProcess.add_significance(x_s, x_e, height_basic + height_add, res[2], axes.flat[plt_count])
                    height_add += 15

            plt_count += 1
            # one_way_anova(group_data_plt)


        data_frame['average'] = data_frame[['mental', 'physical', 'temporal', 'performance','effort', 'frustration']].mean(axis=1)
        group_data = []
        for j in range(group_count):
            group_data.append((data_frame.iloc[(group_judge == j).values, data_frame.columns.get_loc('average')]))
        print("------" + "average result:")
        p_value = Toolbox.fried_man_test(group_data)[1]
        if p_value < 0.06:
            sig_group = Toolbox.wilcoxon_post_hoc(group_data)

        if plot:
            # Adjust height for all plot to maintain same y-axis
            for ax in axes.flat:
                ax.set_ylim(-5, 100 + max_sig_count * 15)
                ax.set_yticks(np.arange(0, 101, 20))
            # fig.tight_layout()
            plt.savefig("NASA_TLX_new.pdf", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

    def count_weight(self, tag):
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



class UnityLogHandler:
    def __init__(self, path, group_names):
        self.path = path
        self.group_names = group_names
        self.df = self.read_jsons()

    def read_jsons(self):
        raw_dict = {'id': [], 'group': []}
        file_list = Toolbox.walk_dir(self.path)
        for file_name in file_list:
            if file_name == '.DS_Store':
                continue
            names = file_name.split('.')[0]
            names = names.split('_')
            # Maybe need to adjust every time (add para in func?)
            raw_dict['id'].append(int(names[1]))
            raw_dict['group'].append(int(names[2]))
            with open(os.path.join(self.path, file_name), 'r') as file:
                data = json.load(file)
                # Save the json data to dict
                for key in data.keys():
                    if key not in raw_dict:
                        raw_dict[key] = []
                    raw_dict[key].append(float(data[key]))
        return pd.DataFrame(raw_dict)

    def analyze_jsons(self, plot_design=(1, 1), start=2, plot=True):
        group_judge = self.df.loc[:, "group"]  # get group number
        plt_count = 0
        max_sig_count = 0
        if plot:
            fig, axes = plt.subplots(plot_design[0], plot_design[1], layout="constrained")

        for index_depend in range(start, self.df.shape[1]):
            group_data = []
            for j in range(len(self.group_names)):
                # TODO: Mind here, deal with nan or not
                group_data.append(self.df.iloc[(group_judge == j).values, index_depend])
            if plot:
                bp_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']
                bp = axes.flat[plt_count].boxplot(group_data, patch_artist=True, labels=self.group_names,
                                                  showfliers=True, showmeans=True)
                # TODO: Create father class and arrange the plot as func
                for patch, color in zip(bp["boxes"], bp_colors):
                    patch.set_facecolor(color)
                    patch.set_linewidth(0.5)
                    # patch.set_edgecolor('white')
                for flier in bp["fliers"]:
                    flier.set(marker='*')
                for mean in bp["means"]:
                    mean.set_markerfacecolor("#86C166")
                    mean.set_markeredgecolor("#86C166")
                axes.flat[plt_count].set_title(self.df.columns[index_depend])
            print("------" + self.df.columns[index_depend] + " result:")
            # 考虑把这一部分改到main里面
            if index_depend == self.df.columns.get_loc('recGoodsFind'):
                p_value = Toolbox.fried_man_test(group_data[1:])[1]
            elif index_depend == self.df.columns.get_loc('adapt.'):
                Toolbox.normal_distribute(group_data[1:])
                p_value = Toolbox.one_anova(group_data[1:])[1]
            elif index_depend == self.df.columns.get_loc('expTime') or index_depend == self.df.columns.get_loc('freq.'):
                Toolbox.normal_distribute(group_data)
                p_value = Toolbox.fried_man_test(group_data)[1]
            else:
                p_value = Toolbox.fried_man_test(group_data)[1]
            if p_value < 0.06:
                sig_group = Toolbox.wilcoxon_post_hoc(group_data)
                max_sig_count = max(max_sig_count, len(sig_group))
                height_add = 0
                height_basic = axes.flat[plt_count].get_ylim()[1]
                # axes.flat[plt_count].set_ylim(-5, height_basic + len(sig_group) * 15)
                # axes.flat[plt_count].set_yticks(np.arange(0, 101, 20))
                for res in sig_group:
                    # Get x_axis positions for start and end
                    x_s = axes.flat[plt_count].get_xticks()[res[0]]
                    x_e = axes.flat[plt_count].get_xticks()[res[1]]
                    NasaTLXProcess.add_significance(x_s, x_e, height_basic + height_add, res[2], axes.flat[plt_count])
                    height_add += 15

            plt_count += 1
            # one_way_anova(group_data_plt)

        if plot:
            # Adjust height for all plot to maintain same y-axis
            # for ax in axes.flat:
                # ax.set_ylim(-5, 100 + max_sig_count * 15)
                # ax.set_yticks(np.arange(0, 101, 20))
            # fig.tight_layout()
            # plt.savefig("unity_log.pdf", dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()



if __name__ == '__main__':
    """
    path = os.path.expanduser("~/Developer/Exp_Result/finished")
    tmp = UnityLogHandler(path, ["NoRS", "Arr.", "High.", "Swap"])
    adjusted_df = tmp.df
    adjusted_df['freq.'] = adjusted_df['newGoodsFind'] / adjusted_df['expTime']
    group_judge = adjusted_df.loc[:, "group"]
    adjusted_df.iloc[(group_judge == 0).values, adjusted_df.columns.get_loc('recGoodsFind')] = 0
    adjusted_df['adapt.'] = adjusted_df['recGoodsFind'] / adjusted_df['newGoodsFind']
    group_lists = adjusted_df['group'].tolist()
    for index, group_name in enumerate(group_lists):
        if group_name == 2:
            group_lists[index] = 3
        elif group_name == 3:
            group_lists[index] = 2
    adjusted_df['group'] = pd.Series(group_lists)
    tmp.df = adjusted_df
    tmp.analyze_jsons((2, 3))
    """
    m_path_NASA = os.path.expanduser("~/Developer/Exp_Result/NASA_TLX")
    nasa_handler = NasaTLXProcess(m_path_NASA, ["NoRS", "Arr.", "High.", "Swap"])
    nasa_handler.read_nasa()

