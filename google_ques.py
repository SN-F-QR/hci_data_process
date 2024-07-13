import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from tool import Toolbox
import numpy as np
from log_data import DataProcess
from collections import Counter


# The current class mainly support within-subject study using one ques
class GoogleQuesProcess(DataProcess):
    def __init__(self, path, group_names, saved_name='google_ques_output.pdf'):
        super().__init__(path, group_names, saved_name)  # Use the path of csv file
        self.mean = pd.Series

    # assume that all ques have their abbr. on the head of ques
    # EXP: 'BC1 - What is your name?' => 'BC1'
    def read_clean_column_names(self, split_char, saved=False):
        data = pd.read_csv(self.path)
        origin_column = data.columns.tolist()[:]
        for index, name in enumerate(origin_column):
            names = name.split(split_char)
            origin_column[index] = names[0]
        data.columns = origin_column
        self.df = data

        if saved:
            data.to_csv('abbr_google_ques.csv', index=False)

    def plot_bar(self, start, amount, mean_ques=True, subplot_titles=None, group_colors=None, fig_size=None, p_correction=False):
        # self.mean = self.df.iloc[:, start:start+amount*self.group_num].mean(numeric_only=True)
        res_mean = [[] for _ in range(self.group_num)]
        res_std = [[] for _ in range(self.group_num)]
        for i in range(self.group_num):
            # use the grouped index to calculate the mean and std
            ques_index = self.split_ques_type(start+amount*i, amount, use_head=mean_ques)
            for _type in ques_index.keys():
                selected_group = self.df.iloc[:, start+amount*i:start+amount*(i+1)]
                averages = selected_group.loc[:, ques_index[_type]].mean(axis=1)
                res_mean[i].append(averages.mean())
                res_std[i].append(averages.std())

        fig, ax = plt.subplots(figsize=fig_size)

        plt.grid(axis='y')
        ax.set_axisbelow(True)
        ind = np.arange(len(ques_index))
        width = 0.81 / self.group_num

        if group_colors is None:
            group_colors = ['#ADD8E6', '#FFDAB9', '#E6E6FA', "#F1A7B5", '#F5F5DC']

        for index, (m, s) in enumerate(zip(res_mean, res_std)):
            ax.bar(ind+index*width, m, width, bottom=0, yerr=s,
                   color=group_colors[index], label=self.group_names[index],
                   error_kw=dict(ecolor='black', lw=1, capsize=2, capthick=1))

        if subplot_titles is None:
            subplot_titles = ques_index.keys()
        ax.set_xticks(ind + width / 2 * (self.group_num - 1), labels=subplot_titles)
        ax.legend()
        ax.autoscale_view()
        plt.show()

    # Return type_dict, where columns head name or full name as key, the full name as value
    def split_ques_type(self, start, amount, use_head=True):
        types = self.df.columns.tolist()[start:start+amount]
        type_dict = {}
        for index, name in enumerate(types):
            if use_head:
                _type = re.findall(r'[a-zA-Z]+', name)[0]
            else:
                _type = name

            if _type not in type_dict:
                type_dict[_type] = [name]
            else:
                type_dict[_type].append(name)
        return type_dict








if __name__ == '__main__':
    howMuchQuesPerGroup = 15
    validQuesStartFrom = 13
    path = os.path.expanduser("~/Developer/Exp_Result/VR Rec Questionnaire.csv")
    google_handler = GoogleQuesProcess(path, ["Group A", "Group B", "Group C"])
    google_handler.read_clean_column_names(' - ')
    google_handler.df = google_handler.df.rename(columns={'TR2': 'IT1'})  # Adjust some unintended columns
    google_handler.plot_bar(validQuesStartFrom, howMuchQuesPerGroup, mean_ques=False, fig_size=(12, 4))



    print('1')

