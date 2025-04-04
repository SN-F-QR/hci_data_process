import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.multicomp import multipletests


class Toolbox:
    @staticmethod
    def walk_dir(path, deep=False):
        # return all file names, if deep, include files in subdir
        def get_valid_files(files):
            return [f for f in files if not f.startswith(".")]

        file_list = []
        temp = os.walk(path)
        for cur_path, dirs, files in temp:
            if deep:
                if cur_path == path:
                    file_list.extend(get_valid_files(files))
                    continue
                directory = cur_path.split("/")[-1]
                for file in files:
                    if not file.startswith("."):
                        file_list.append(os.path.join(directory, file))
            else:
                file_list = get_valid_files(files)
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
                print("Group", i, "vs", j, ": statistic =", p[0], "p =", p[1])
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
            if min(all_p_values) < 0.05:
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
    def auto_pair_significance_test(data_group):
        """
        :param data_group: the list contains data arranged by groups like [[], []]
        :reutrn: p value
        """
        normal_distributed = Toolbox.normal_distribute(data_group, is_list=True)

        if normal_distributed:
            result = stats.ttest_rel(data_group[0], data_group[1])
            print("Normal Distributed, use paired T-test")
            print("T-test result:", result)
        else:
            result = stats.wilcoxon(data_group[0], data_group[1])
            print("Not Normal Distributed, use Wilcoxon test")
            print(f"Wilcoxon result: statistic = {result[0]}, p = {result[1]}")

        return result[1]

    @staticmethod
    def auto_multiple_significance_test(data_group):
        sig_group = []

        if Toolbox.normal_distribute(data_group):
            print("Normal Distributed, use One-way ANOVA test")
            p_value = Toolbox.one_anova(data_group)[1]
            if p_value < 0.06:
                sig_group = Toolbox.tukey_post_hoc(data_group)
        else:
            print("Not Normal Distributed, use Friedman test")
            p_value = Toolbox.fried_man_test(data_group)[1]
            if p_value < 0.06:
                sig_group = Toolbox.wilcoxon_post_hoc(data_group, bonferroni_holm=False)
        return sig_group

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
