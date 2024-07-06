import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tool import Toolbox
from main import NasaTLXProcess

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
                p_value = Toolbox.one_anova(group_data)[1]
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

