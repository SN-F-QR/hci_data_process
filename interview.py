import os
import pandas as pd
from tool import Toolbox
from dotenv import load_dotenv


class InterviewDataProcess:
    def __init__(self, path, output_path):
        self.path = path
        self.output_path = output_path

    def convert_files(self, handler, endwith):
        path_files = Toolbox.get_files_endwith(self.path, endwith)
        for path, fileNames in path_files:
            for fileName in fileNames:
                cleaned_text = self.clean_time_stamp(path, fileName)
                if self.output_path != "":
                    handler(cleaned_text, self.output_path, fileName)
                else:
                    handler(cleaned_text, path, fileName)

    def create_cleaned_files(self):
        self.convert_files(self.save_cleaned_file, ".txt")

    def create_tagged_table(self):
        self.convert_files(self.save_table, ".txt")

    def clean_time_stamp(self, file_path, file_name):
        path = os.path.join(file_path, file_name)
        with open(
            path, "r", encoding="utf-8", errors="surrogateescape"
        ) as file:  # errors to handle non-utf-8 characters, keep original content
            lines = file.readlines()
            cleaned_lines = map(lambda line: line.strip().split("]")[-1].strip(), lines)
            return cleaned_lines

    def classify_to_table(self, tagged_lines):
        """
        Convert the txt with tag to a csv table.
        :return: one column csv table
        :example: ["1", "Ask a question", "2", "Ans1", "Ans1.5" "-", "Ans2"], will return ["EX", "Ask a question\n", "SU", "Ans1\nAns1.5\n", "Ans2\n"]
        """
        table = []
        cur_line = ""

        def append_line(line):
            if line != "":
                table.append(line)
            return ""

        for line in tagged_lines:
            if line == "1":
                cur_line = append_line(cur_line)
                table.append("EX")
            elif line == "2":
                cur_line = append_line(cur_line)
                table.append("SU")
            elif line == "-":
                cur_line = append_line(cur_line)
            elif cur_line == "":
                cur_line = line + "\n"
            else:
                cur_line += line + "\n"

        append_line(cur_line)
        tagged_table = pd.Series(table)

        return tagged_table

    def save_table(self, cleaned_tagged_lines, saved_path, fileName):
        path = os.path.join(saved_path, fileName.split(".")[0] + ".csv")
        tagged_table = self.classify_to_table(cleaned_tagged_lines)
        tagged_table.to_csv(path, index=True)

    def save_cleaned_file(self, cleaned_lines, saved_path, fileName):
        path = os.path.join(saved_path, "cleaned_" + fileName)
        with open(path, "w") as s_file:
            for line in cleaned_lines:
                s_file.write(line + "\n")


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("INTERVIEW_PATH")
    interview_data = InterviewDataProcess(path, os.getenv("OUTPUT_PATH"))
    interview_data.create_tagged_table()
