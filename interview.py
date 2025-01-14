import os
from dotenv import load_dotenv


class InterviewDataProcess:
    def __init__(self, path):
        self.path = path

    def create_cleaned_files(self):
        temp = os.walk(self.path, topdown=True)
        for path, dirs, files in temp:
            txt_files = list(filter(lambda name: name.endswith(".txt"), files))
            for txt in txt_files:
                self.clean_time_stamp(path, txt)

    def clean_time_stamp(self, file_path, file_name):
        path = os.path.join(file_path, file_name)
        saved_path = os.path.join(file_path, "cleaned_" + file_name)
        with open(
            path, "r", encoding="utf-8", errors="surrogateescape"
        ) as file:  # errors to handle non-utf-8 characters, keep original content
            lines = file.readlines()
            cleaned_lines = map(lambda line: line.strip().split("]")[-1].strip(), lines)
            with open(saved_path, "w") as s_file:
                for line in cleaned_lines:
                    s_file.write(line + "\n")


if __name__ == "__main__":
    load_dotenv()
    path = os.getenv("INTERVIEW_PATH")
    interview_data = InterviewDataProcess(path)
    interview_data.create_cleaned_files()
