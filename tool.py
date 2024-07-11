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
        return file_list

    @staticmethod
    def fried_man_test(data_group):
        p = stats.friedmanchisquare(*data_group)
        print("Friedman: The null hypothesis cannot be rejected when p>0.05:", p)
        # p = stats.kruskal(data_group[0], data_group[1], data_group[2])
        # p = stats.median_test(data_group[0], data_group[1], data_group[2], nan_policy='omit')
        return p

    @staticmethod
    def wilcoxon_post_hoc(data_group, bonferroni_holm=False):
        # can use `from itertools import combinations`
        significant = []
        print("Found significant difference, run wilcoxon post-hoc test")
        print("Wilcoxon: Reject the null hypothesis that there is no difference when p<0.05")
        p_group = []
        for i in range(len(data_group)):
            for j in range(min(i + 1, len(data_group)), len(data_group)):
                p = stats.wilcoxon(data_group[i], data_group[j], correction=False, method="auto",
                                   alternative="two-sided", nan_policy="omit")
                print("Group", i, "vs", j, ":", p)
                p_group.append(p[1])
                if p.pvalue <= 0.05 or bonferroni_holm:
                    significant.append((i, j, p.pvalue))
        if bonferroni_holm:
            print("----------Result after Bonferroni-Holm correction----------")
            reject, p_adjusted, _, _ = multipletests(p_group, method='holm')
            print("Reject null?:", reject, "Adjusted p:", p_adjusted)
            final_sig = []
            for i, j, p, state, p_new in zip(significant, reject, p_adjusted):
                if state:
                    final_sig.append((i, j, p_new))
            significant = final_sig
        return significant

    @staticmethod
    def normal_distribute(data_group, is_list=True):
        if is_list:
            for one_group in data_group:
                Toolbox.normal_distribute(one_group, False)
        else:
            stat, p = stats.shapiro(data_group)
            print('The normal distribution outputs p:', p, 'with stats:', stat)

    @staticmethod
    def one_anova(data_group):
        stat, p = stats.f_oneway(*data_group)
        print("One-way ANOVA: The null hypothesis cannot be rejected when p>0.05:", p)
        return [stat, p]
