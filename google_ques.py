import os
import pandas as pd
from tool import Toolbox


class GoogleQues:
    def __init__(self, path, group_names):
        self.data = []
        self.path = path
        self.group_names = group_names
        self.file_names = []

        # plot
        self.plots = []

    def clean_column_names(self, saved=False):
        data = pd.read_csv(self.path)
        origin_column = data.columns.tolist()
        new_column = origin_column[:]
        for index, name in enumerate(origin_column):
            names = name.split(' - ')
            new_column[index] = names[0]
        data.columns = new_column
        self.data = data

        if saved:
            data.to_csv('cleaned_ques.csv', index=False)


    def split_csv(self, start, length, repeat_time):
        return









if __name__ == '__main__':
    path = os.path.expanduser("~/Developer/Exp_Result/VR Rec Questionnaire.csv")
    tmp = GoogleQues(path, ["NoRS", "Arr.", "Swap", "High."])
    tmp.clean_column_names(True)