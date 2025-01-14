import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests


class Toolbox:
    @staticmethod
    def walk_dir(path):
        # return all file names
        file_list = []
        temp = os.walk(path)
        for path, dirs, files in temp:
            file_list = files
        file_list = [f for f in file_list if not f.startswith(".")]
        return file_list

    @staticmethod
    def get_files_endwith(path, end):
        temp = os.walk(path, topdown=True)
        target_files = []
        for path, dirs, files in temp:
            target_files.append(
                (path, list(filter(lambda name: name.endswith(end), files)))
            )
        return target_files

    @staticmethod
    def fried_man_test(data_group):
        """
        Conduct friedman significant test
        :param data_group: the list contains data arranged by groups, 3 groups like [[], [], []]
        :return p: the p value for all groups
        """
        p = stats.friedmanchisquare(*data_group)
        print("Friedman: The null hypothesis cannot be rejected when p>0.05:", p)
        # p = stats.kruskal(data_group[0], data_group[1], data_group[2])
        # p = stats.median_test(data_group[0], data_group[1], data_group[2], nan_policy='omit')
        return p

    @staticmethod
    def wilcoxon_post_hoc(data_group, bonferroni_holm=False):
        """
        Conduct wilcoxon_post_hoc significant test
        Compare and calculate between each two group one time
        :param data_group: the list contains data arranged by groups, 3 groups like [[], [], []]
        :param bonferroni_holm: if true, will use bonferroni_holm correction
        :return significant: tuple (i,j,p), where i/j are the two groups with significant differece of p value
        """
        # can use `from itertools import combinations`
        significant = []
        print("Found significant difference, run wilcoxon post-hoc test")
        print(
            "Wilcoxon: Reject the null hypothesis that there is no difference when p<0.05"
        )
        p_group = []
        for i in range(len(data_group)):
            for j in range(min(i + 1, len(data_group)), len(data_group)):
                p = stats.wilcoxon(
                    data_group[i],
                    data_group[j],
                    correction=False,
                    method="auto",
                    alternative="two-sided",
                    nan_policy="omit",
                )
                print("Group", i, "vs", j, ":", p)
                p_group.append(p[1])
                if p.pvalue <= 0.05 or bonferroni_holm:
                    significant.append((i, j, p.pvalue))
        if bonferroni_holm:
            print("----------Result after Bonferroni-Holm correction----------")
            reject, p_adjusted, _, _ = multipletests(p_group, method="holm")
            print("Reject null?:", reject, "Adjusted p:", p_adjusted)
            final_sig = []
            for sig, state, p_new in zip(significant, reject, p_adjusted):
                if state:
                    final_sig.append((sig[0], sig[1], p_new))
            significant = final_sig
        return significant

    @staticmethod
    def normal_distribute(data_group, index=-1, is_list=True):
        if is_list:
            all_p_values = []
            print("Wilk: The null hypothesis cannot be rejected when p>0.05:")
            for g_index, one_group in enumerate(data_group):
                all_p_values.append(
                    Toolbox.normal_distribute(one_group, index=g_index, is_list=False)
                )
            if max(all_p_values) < 0.05:
                return False
            else:
                return True
        else:
            stat, p = stats.shapiro(data_group)
            print(
                "Group",
                index,
                "The normal distribution outputs p:",
                p,
                "with stats:",
                stat,
            )
            return p

    @staticmethod
    def one_anova(data_group):
        stat, p = stats.f_oneway(*data_group)
        print("One-way ANOVA: The null hypothesis cannot be rejected when p>0.05:", p)
        return [stat, p]

    @staticmethod
    def tukey_post_hoc(data_group):
        print("Found significant difference, run Tukey's post-hoc test")
        post_result = stats.tukey_hsd(*data_group)
        stat_group = post_result.statistic
        p_values = post_result.pvalue
        p_group = []
        print(post_result)
        for index, value in np.ndenumerate(stat_group):
            i, j = index
            if i > j:
                continue  # Skip the repeated pair
            p = p_values[i, j]
            if p <= 0.05:
                p_group.append((i, j, p_values[i, j]))
        return p_group
