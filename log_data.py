import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool import Toolbox
import warnings


class DataProcess:
    def __init__(self, path, group_names, saved_name='output.pdf'):
        self.path = path  # path to the document includes data file(s)
        self.group_names = group_names  # experimental groups
        self.saved_name = saved_name
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
    def plot_sub_data(self, start=2, fig_design=(2, 3), subplot_titles=None, group_colors=None, flier_mark='o', p_corretion=False):
        if group_colors is None:
            group_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']
        assert self.df["group"].nunique() == self.group_num
        # plt_count = 0
        max_sig_count = 0  # To decide the max height of each subplot
        fig, axes = plt.subplots(fig_design[0], fig_design[1], layout="constrained")
        self.plots = axes
        # fig, axes = plt.subplots(1, 6, layout="constrained", figsize=(12, 4)) # for two columns
        if fig_design[0] * fig_design[1] < self.df.shape[1] - start:
            end = fig_design[0] * fig_design[1] + start
            warnings.warn('Variables after ' + self.df.columns[end-1] + ' will be ignored in subplots.')
        else:
            end = self.df.shape[1]

        for i in range(start, end):  # column index, namely the dependent variables
            group_data = self.extract_by_group(i)
            plt_index = i - start

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
            sig_group = self.significance_test(self.df.columns[i], group_data, p_corretion)
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
        plt.savefig(self.saved_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # arrange the data by group in using the index of column
    def extract_by_group(self, column_index):
        group_judge = self.df.loc[:, "group"]  # get group number for filtering
        group_data = []
        for j in range(self.group_num):
            # group_data.append(delete_outlier(data_frame.iloc[(group_judge == j).values, i]))
            group_data.append((self.df.iloc[(group_judge == j).values, column_index]))
        return group_data

    # apply a new value to one group, use to clean unintended data
    def apply_by_group(self, apply_value, apply_group, column_index):
        group_judge = self.df.loc[:, "group"]
        self.df.iloc[(group_judge == apply_group).values, column_index] = apply_value
        print('All', self.df.columns[column_index], 'in group', apply_group, 'set to', apply_value)

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


class TLXProcess(DataProcess):
    # Judge Raw or Weighted automatically, use raw_nasa to use raw only
    def __init__(self, path, group_names, saved_name='nasa_tlx_output.pdf', raw_nasa=False):
        super().__init__(path, group_names, saved_name)
        self.rs_data = pd.DataFrame()
        self.pw_data = pd.DataFrame()
        self.raw_nasa = raw_nasa
        self.read_nasa()

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
        self.df = self.rs_data
        # get raw result
        print("---NASA TLX raw result---")
        # get weighted result
        if weighted and not self.raw_nasa:
            # read pair result and get weight
            pw_frame = self.read_nasa_raw('pw')
            print("---NASA TLX weighted result---")
            self.pw_data = self.rs_data[:]  # avoid ref
            self.pw_data.loc[:, "mental":] = self.rs_data.loc[:, "mental":] * pw_frame.loc[:, "mental":] / 15
            self.df = self.pw_data

        # weighted_frame.loc[:, "sum"] = weighted_frame.iloc[:, 2]
        # for i in range(3,8):
        # weighted_frame.loc[:, "sum"] += weighted_frame.iloc[:, i]
        # return 6*2 results

    def read_nasa_raw(self, raw_type):
        assert raw_type == 'pw' or raw_type == 'rs'
        # dict for dataframe
        data_dict = {"id": [],
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
                        data_dict["id"].append(int(words[1]))
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

    def plot_sub_data(self, **kwargs):
        super().plot_sub_data(**kwargs)

    # TODO: Add plot option
    def nasa_average(self, start=2):
        self.df['average'] = self.df.iloc[:, start:start+6].mean(axis=1)
        average_by_group = self.extract_by_group(self.df.columns.get_loc('average'))
        self.significance_test('Overall TLX', average_by_group)


# Handle Objective data in Unity Json
# Assume that file name is like: expName_participantNo._groupNAME_xxx
class UnityJsonProcess(DataProcess):
    def __init__(self, path, group_names, saved_name='unity_json_output.pdf'):
        super().__init__(path, group_names, saved_name)

    def read_jsons(self, where_id=1, where_group=2):
        raw_dict = {'id': [], 'group': []}
        file_list = Toolbox.walk_dir(self.path)
        for file_name in file_list:
            names = file_name.split('.')[0]
            names = names.split('_')
            raw_dict['id'].append(int(names[where_id]))
            raw_dict['group'].append(int(names[where_group]))
            with open(os.path.join(self.path, file_name), 'r') as file:
                data = json.load(file)
                # Save the json data to dict
                for key in data.keys():
                    if key not in raw_dict:
                        raw_dict[key] = []
                    raw_dict[key].append(float(data[key]))
        self.df = pd.DataFrame(raw_dict)

    def plot_sub_data(self, start=2, fig_design=(2, 3), subplot_titles=None, group_colors=None, flier_mark='o', p_corretion=False):
        super().plot_sub_data()



if __name__ == '__main__':
    group_names = ["A", "B", "C", "D"]
    # plot_titles = ['time', 'behavior1', 'behavior2', 'behavior3', 'behavior4', 'behavior5']

    path = os.path.expanduser("~/Developer/Exp_Result/finished")
    unity_handler = UnityJsonProcess(path, group_names)
    unity_handler.read_jsons(where_id=1, where_group=2)
    unity_handler.apply_by_group(0, 0, unity_handler.df.columns.get_loc('recGoodsFind'))
    adjusted_df = unity_handler.df
    adjusted_df['freq.'] = adjusted_df['newGoodsFind'] / adjusted_df['expTime']
    adjusted_df['adapt.'] = adjusted_df['recGoodsFind'] / adjusted_df['newGoodsFind']
    unity_handler.df = adjusted_df
    unity_handler.plot_sub_data(start=2, fig_design=(2, 3),
                                group_colors=None, p_corretion=False)


    """
    m_path_NASA = os.path.expanduser("~/Developer/Exp_Result/NASA_TLX")
    plot_titles = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    nasa_handler = TLXProcess(m_path_NASA, ["A", "B", "C", "D"], raw_nasa=False)
    nasa_handler.read_nasa()
    nasa_handler.nasa_average(start=2)  # Calculate average score and add to dataframe
    nasa_handler.plot_sub_data(start=2, fig_design=(2, 3),
                               subplot_titles=plot_titles, group_colors=None, p_corretion=False)
    """