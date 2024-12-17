import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool import Toolbox
import warnings


class DataProcess:
    def __init__(self, path, group_names, saved_name="output.pdf", group_colors=None):
        """
        Initialize class

        :param path: path to the document includes data file(s)
        :param group_names: the names of experimental groups
        :param saved_name: the name for saved plot using pdf format
        :param group_colors: the colors related to each experimental groups, based on hexadecimal
        """
        self.path = path
        self.group_names = group_names
        self.saved_name = saved_name
        self.group_num = len(group_names)
        self.df = pd.DataFrame()

        self.plots = []  # record all plots in matplot for customization
        self.fig = plt.figure()
        if group_colors is None:
            group_colors = ["#ADD8E6", "#FFDAB9", "#E6E6FA", "#F1A7B5", "#F5F5DC"]
        self.group_colors = group_colors

    # Assume that file name is like: expName_groupNAME_participantNo._xxx
    # def read_data(self, where_group=1, where_user=2):

    def plot_sub_data(
        self,
        start=2,
        fig_design=(2, 3),
        fig_size=(6, 5),
        subplot_titles=None,
        flier_mark="o",
        same_yaxis=None,
        p_correction=False,
    ):
        """
        Use subplots to show box plot of the scores.
        Will update self.fig and self.plots

        :param start: the first column index of useful data
        :param fig_design: should be tuple, like (2,3) => 2 x 3 subplot design
        :param fig_size: should be tuple, the output size (in inch) of plot
        :param subplot_titles: titles for each subplot
        :param flier_mark: string mark for outlier, same with matplot
        :param same_yaxis: if true, all subplots use the same provided y-axis
        :param p_correction: if true, use Bonferroni-Holm correction in post-hoc analyze
        """
        assert self.df["group"].nunique() == self.group_num
        # plt_count = 0
        max_sig_count = 0  # To decide the max height of each subplot
        fig, axes = plt.subplots(
            fig_design[0], fig_design[1], layout="constrained", figsize=fig_size
        )
        self.plots = axes
        self.fig = fig
        # fig, axes = plt.subplots(1, 6, layout="constrained", figsize=(12, 4)) # for two columns
        if fig_design[0] * fig_design[1] < self.df.shape[1] - start:
            end = fig_design[0] * fig_design[1] + start
            warnings.warn(
                "Variables after "
                + self.df.columns[end - 1]
                + " will be ignored in subplots."
            )
        else:
            end = self.df.shape[1]

        for i in range(start, end):  # column index, namely the dependent variables
            group_data = self.extract_by_group(i)
            plt_index = i - start

            # Subplot Setting
            # supported color
            # bp_colors = ["#5184B2", "#AAD4F8", "#F1A7B5", "#D55276", "#F2F5FA"]
            bp = axes.flat[plt_index].boxplot(
                group_data,
                patch_artist=True,
                labels=self.group_names,
                showfliers=True,
                showmeans=True,
            )

            for patch, color in zip(bp["boxes"], self.group_colors):
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
            sig_group = self.significance_test(
                self.df.columns[i], group_data, p_correction
            )
            if len(sig_group) > 0:
                max_sig_count = max(max_sig_count, len(sig_group))
                height_add = 0
                height_min, height_basic = axes.flat[plt_index].get_ylim()
                height_diff = (height_basic - height_min) / 6
                axes.flat[plt_index].set_ylim(
                    height_min, height_basic + len(sig_group) * height_diff
                )
                # axes.flat[plt_index].set_yticks(np.arange(0, 101, 20))
                for res in sig_group:
                    # Get x_axis positions for start and end
                    x_s = axes.flat[plt_index].get_xticks()[res[0]]
                    x_e = axes.flat[plt_index].get_xticks()[res[1]]
                    self.add_significance(
                        x_s,
                        x_e,
                        height_basic + height_add,
                        res[2],
                        axes.flat[plt_index],
                        height_diff / 4,
                    )
                    height_add += height_diff

        # Adjust height for all plot to maintain same y-axis
        if same_yaxis is not None:
            height_min, height_max = 0, 0
            for ax in axes.flat:
                height_min = min(height_min, ax.get_ylim()[0])
                height_max = max(height_max, ax.get_ylim()[1])
            for ax in axes.flat:
                ax.set_ylim(height_min, height_max)
                ax.set_yticks(same_yaxis)
        # fig.tight_layout()
        # plt.savefig(self.saved_name, dpi=300, bbox_inches='tight')
        self.fig.show()
        # plt.close()

    def set_sub_yticks(self, sub_index, y_range):
        self.plots.flat[sub_index].set_yticks(y_range)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.show()

    def save_fig(self):
        self.fig.savefig(self.saved_name, dpi=300, bbox_inches="tight")

    def extract_by_group(self, column_index):
        """
        Return the data arranged by group in the index of column
        The group name must be Int from 0
        :param column_index: the column index for dependent variables
        :return group_data: [[group1], [group2], ...]
        """
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
        print(
            "All",
            self.df.columns[column_index],
            "in group",
            apply_group,
            "set to",
            apply_value,
        )

    # Re-write this if other analyze method is expected
    def significance_test(self, name, data, correction=False):
        """
        Deploy significant test using friedman and wilcoxon post test

        :param name: the name of tested type of objective data in json
        :param data: the list contains objective data arranged by groups, 3 groups like [[], [], []]
        :param correction: same with p_correction in plot_sub_data
        :return sig_group: tuple (i,j,p), where i/j are the two groups with significant difference of p value
        """
        print("-------" + name + " result:")
        p_value = Toolbox.fried_man_test(data)[1]
        sig_group = []
        # include borderline condition
        if p_value < 0.06:
            sig_group = Toolbox.wilcoxon_post_hoc(data, bonferroni_holm=correction)

        return sig_group

    @staticmethod
    def add_significance(start, end, height, p_value, ax, scale):
        """
        Draw significant line and sign in plot

        :param start: left x position in plot
        :param end: right x position in plot
        :param height: y position of line
        :param p_value: p value of significant test
        :param ax: axis of plot
        :param scale: short line in significant line and the height of significant signs
        """
        x = [start, start, end, end]
        y = [height, height + scale, height + scale, height]
        ax.plot(x, y, color="k", linewidth=1.5)

        sign = ""
        if p_value < 0.001:
            sign = "***"
        elif p_value < 0.01:
            sign = "**"
        elif p_value <= 0.05:
            sign = "*"

        ax.text((start + end) / 2, height + scale, sign, ha="center", va="bottom")


class TLXProcess(DataProcess):
    def __init__(
        self, path, group_names, saved_name="nasa_tlx_output.pdf", raw_nasa_only=False
    ):
        """
        :param raw_nasa_only: if true, will only load files of raw nasa-tlx
        """
        super().__init__(path, group_names, saved_name)
        self.file_names = Toolbox.walk_dir(self.path)  # record all data files
        self.rs_data = pd.DataFrame()
        self.pw_data = pd.DataFrame()
        self.raw_nasa = raw_nasa_only

    def read_nasa(self):
        """
        Read and load NASA-TLX files, basically judge Raw or Weighted automatically
        Only handle NASA_TLX with or without each-time pairwise
        Return result in self.rs_data and self.pw_data
        """
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
        self.rs_data = self.read_nasa_raw("rs")
        self.df = self.rs_data
        # get raw result
        print("---NASA TLX raw result---")
        # get weighted result
        if weighted and not self.raw_nasa:
            # read pair result and get weight
            pw_frame = self.read_nasa_raw("pw")
            print("---NASA TLX weighted result---")
            self.pw_data = self.rs_data[:]  # avoid ref
            self.pw_data.loc[:, "mental":] = (
                self.rs_data.loc[:, "mental":] * pw_frame.loc[:, "mental":] / 15
            )
            self.df = self.pw_data

        # weighted_frame.loc[:, "sum"] = weighted_frame.iloc[:, 2]
        # for i in range(3,8):
        # weighted_frame.loc[:, "sum"] += weighted_frame.iloc[:, i]
        # return 6*2 results

    def read_nasa_raw(self, raw_type):
        """
        Load raw score files or pairwise score files
        :param raw_type: string, must be 'pw' to load pairwise score or 'rs' to load raw score
        :return raw_frame: dataframe of scores including participants id / group / scores
        """
        assert raw_type == "pw" or raw_type == "rs"
        # dict for dataframe
        data_dict = {
            "id": [],
            "group": [],
            "Mental Demand": [],
            "Physical Demand": [],
            "Temporal Demand": [],
            "Performance": [],
            "Effort": [],
            "Frustration": [],
        }

        for name in self.file_names:
            full_path = os.path.join(self.path, name)
            with open(full_path, "r") as csv_txt:
                data_flag = False
                weight = {
                    "Mental Demand": 0,
                    "Physical Demand": 0,
                    "Temporal Demand": 0,
                    "Performance": 0,
                    "Effort": 0,
                    "Frustration": 0,
                }  # store weight for pw and score for raw
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
                        if raw_type == "pw":
                            weight[words[-1].replace("\n", "")] += 1
                        elif raw_type == "rs":
                            weight[words[0]] = int(words[-1])

                for key in weight.keys():
                    data_dict[key].append(weight[key])

        raw_frame = pd.DataFrame(data_dict)
        return raw_frame

    def plot_sub_data(self, **kwargs):
        super().plot_sub_data(**kwargs)

    def nasa_average(self, start=2):
        """
        Calculate the overall NASA-TLX scores, typically ignoring plot
        And provide significant tests
        :param start: the first column index of useful data
        """
        self.df["average"] = self.df.iloc[:, start : start + 6].mean(axis=1)
        average_by_group = self.extract_by_group(self.df.columns.get_loc("average"))
        self.significance_test("Overall TLX", average_by_group)


# Handle Objective data in Unity Json
# Assume that file name is like: expName_participantNo._groupNAME_xxx
class UnityJsonProcess(DataProcess):
    def __init__(self, path, group_names, saved_name="unity_json_output.pdf"):
        super().__init__(path, group_names, saved_name)
        self.file_names = Toolbox.walk_dir(self.path)  # record all data files

    def read_jsons(self, where_id=1, where_group=2):
        """
        Read the json files from Unity
        Return dataframe by revising self.df
        :param where_id: the position of participant ID in file name
        :param where_group: the position of experiment group in file name
        """
        raw_dict = {"id": [], "group": []}
        file_list = Toolbox.walk_dir(self.path)
        for file_name in file_list:
            names = file_name.split(".")[0]
            names = names.split("_")
            raw_dict["id"].append(int(names[where_id]))
            raw_dict["group"].append(int(names[where_group]))
            with open(os.path.join(self.path, file_name), "r") as file:
                data = json.load(file)
                # Save the json data to dict
                for key in data.keys():
                    if key not in raw_dict:
                        raw_dict[key] = []
                    raw_dict[key].append(float(data[key]))
        self.df = pd.DataFrame(raw_dict)

    def significance_test(self, name, data, correction=False):
        """
        Deploy significant test by first validating normal distribution
        If normal, use ANOVA and tukey post test
        """
        print("-------Verifying Normal Distribution for " + name)
        if Toolbox.normal_distribute(data):
            p_value = Toolbox.one_anova(data)[1]
            sig_group = []
            if p_value < 0.06:
                sig_group = Toolbox.tukey_post_hoc(data)
            return sig_group
        else:
            return super().significance_test(
                name, data, correction
            )  # You may want other significant test
