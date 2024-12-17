import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from tool import Toolbox
import numpy as np
from log_data import DataProcess


# The current class mainly support within-subject study using one ques
class GoogleQuesProcess(DataProcess):
    def __init__(self, path, group_names, saved_name="google_ques_output.pdf"):
        """
        :param path: the path to the csv file
        """
        super().__init__(path, group_names, saved_name)
        self.mean = pd.Series

    def read_clean_column_names(self, split_char, saved=False):
        """
        Read the scores from csv
        Refine the name for each question, assume that all ques have their abbr. on the head of ques
            'BC1 - What is your name?' => 'BC1'
        Args:
            split_char: the split char between abbr. and the question
            saved: if true, saved the new csv with abbr.
        Return scores as dataframe by revising self.df
        """
        data = pd.read_csv(self.path)
        origin_column = data.columns.tolist()[:]
        for index, name in enumerate(origin_column):
            names = name.split(split_char)
            origin_column[index] = names[0]
        data.columns = origin_column
        self.df = data

        if saved:
            data.to_csv("abbr_google_ques.csv", index=False)

    # this plot assumes that the ques used N point likert scale
    def plot_bar(
        self,
        start,
        amount,
        mean_ques=True,
        subplot_titles=None,
        fig_size=None,
        draw_mean=True,
        p_correction=False,
    ):
        """
        Draw a bar plot for questionnaire scores

        :param start: the first column index of useful data
        :param amount: the number of ques in one group
        :param fig_size: should be tuple, the output size (in inch) of plot
        :param subplot_titles: titles for each subplot
        :param mean_ques: if true, will combine all ques and calculate a mean value for the same questionnaire
        :param draw_mean: if true, draw the mean text on the top of each bar
        :param p_correction: if true, use Bonferroni-Holm correction in post-hoc analyze
        """
        res_mean = [[] for _ in range(self.group_num)]  # the mean of each ques
        res_std = [[] for _ in range(self.group_num)]  # the std of each ques
        res_mean_per_users = [
            [] for _ in range(self.group_num)
        ]  # the mean of each participant
        for i in range(self.group_num):
            # use the grouped index to calculate the mean and std
            ques_index = self.split_ques_type(
                start + amount * i, amount, use_head=mean_ques
            )
            for _type in ques_index.keys():
                selected_group = self.df.iloc[
                    :, start + amount * i : start + amount * (i + 1)
                ]
                averages = selected_group.loc[:, ques_index[_type]].mean(axis=1)
                res_mean[i].append(averages.mean())
                res_std[i].append(averages.std())
                res_mean_per_users[i].append(
                    averages.tolist()
                )  # saved the original averages for significance ana.

        fig, ax = plt.subplots(figsize=fig_size)
        self.fig = fig
        self.plots = ax
        plt.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ind = np.arange(len(ques_index))
        width = 0.81 / self.group_num

        for index, (m, s) in enumerate(zip(res_mean, res_std)):
            ax.bar(
                ind + index * width,
                m,
                width,
                bottom=0,
                yerr=s,
                color=self.group_colors[index],
                label=self.group_names[index],
                error_kw=dict(ecolor="gray", lw=1, capsize=2, capthick=1),
            )
            # Draw mean on the top of bar
            if draw_mean:
                for x, y in zip(ind, m):
                    ax.text(
                        x + index * width,
                        y,
                        f"{y:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        if subplot_titles is None:
            subplot_titles = ques_index.keys()
        ax.set_xticks(ind + width / 2 * (self.group_num - 1), labels=subplot_titles)

        # Add significant signs
        height_min, height_basic = ax.get_ylim()
        ax.set_yticks(np.arange(height_min, height_basic, 1))
        height_basic -= (
            0.3  # Typically the max height will higher than max score in ques?
        )
        height_diff = 0.6
        for index, _type in enumerate(ques_index.keys()):
            dependent_data = []
            for group in res_mean_per_users:
                dependent_data.append(group[index])
            sig_group = self.significance_test(_type, dependent_data, p_correction)
            height_add = 0
            if len(sig_group) > 0:
                for res in sig_group:
                    x_s = ax.get_xticks()[index] + (res[0] - 1) * width
                    x_e = ax.get_xticks()[index] + (res[1] - 1) * width
                    self.add_significance(
                        x_s, x_e, height_basic + height_add, res[2], ax, height_diff / 4
                    )
                    height_add += height_diff
        ax.legend()
        ax.autoscale_view()
        self.fig.show()

    def split_ques_type(self, start, amount, use_head=True):
        """
        Assume the questions for one experimental condition are continuous in colmun
        Group these questions by their types using a dictionary
            EXP: Q1 Q2 P1 P2 => P:[P1, P2], Q:[Q1, Q2]
        Args:
            start: the first column index of question
            amount: int, the number of questions for each experimental condition
            use_head: if true, using the type of question as key, like P:.. Q:.., else P1: P2:
        Returns:
            type_dict: dictionary, where questions' type or each question as key, the full name as value
        """
        types = self.df.columns.tolist()[start : start + amount]
        type_dict = {}
        for index, name in enumerate(types):
            if use_head:
                _type = re.findall(r"[a-zA-Z]+", name)[0]
            else:
                _type = name

            if _type not in type_dict:
                type_dict[_type] = [name]
            else:
                type_dict[_type].append(name)
        return type_dict
