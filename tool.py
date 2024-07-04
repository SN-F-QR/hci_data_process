import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


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
    def wilcoxon_post_hoc(data_group):
        # can use `from itertools import combinations`
        significant = []
        print("Found significant difference, run wilcoxon post-hoc test")
        print("Wilcoxon: Reject the null hypothesis that there is no difference when p<0.05")
        for i in range(len(data_group)):
            for j in range(min(i + 1, len(data_group)), len(data_group)):
                p = stats.wilcoxon(data_group[i], data_group[j], correction=True, method="approx",
                                   alternative="two-sided",
                                   nan_policy="omit")
                print("Group", i, "vs", j, ":", p)
                if p.pvalue <= 0.05:
                    significant.append((i, j, p.pvalue))
        return significant
