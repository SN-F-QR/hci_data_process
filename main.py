import os
import numpy as np
import log_data as handler
from google_ques import GoogleQuesProcess
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    group_names = ["A", "B", "C", "D"]  # Each Group Name
    plot_titles = [
        "time",
        "behavior1",
        "behavior2",
        "behavior3",
        "behavior4",
        "behavior5",
    ]  # Titles for each subplot
    path = os.path.expanduser(
        os.getenv("UNITY_DATA_PATH")
    )  # Path to folder including data files
    # Draw and Analyze Unity Json Data
    unity_handler = handler.UnityJsonProcess(path, group_names)
    unity_handler.read_jsons(where_id=1, where_group=2)
    # unity_handler.df['freq.'] = adjusted_df['behavior1'] / adjusted_df['time']  # exp. for additional data
    # unity_handler.apply_by_group(0, 0, unity_handler.df.columns.get_loc('behavior1'))  # exp. for clean data
    unity_handler.plot_sub_data(
        start=2,
        fig_design=(1, 6),
        fig_size=(12, 3),
        subplot_titles=plot_titles,
        same_yaxis=None,
        p_correction=False,
    )
    # exp. for adjust the value of y_axis when same_yaxis set False
    # unity_handler.set_sub_yticks(sub_index=1, y_range=np.arange(0, 51, 10))
    unity_handler.save_fig()  # Always to save Figure after all adjustments has conducted

    # Draw and Analyze NASA TLX data
    m_path_NASA = os.path.expanduser(
        os.getenv("NASA_TLX_PATH")
    )  # Path to folder including data files
    plot_titles = [
        "mental",
        "physical",
        "temporal",
        "performance",
        "effort",
        "frustration",
    ]
    nasa_handler = handler.TLXProcess(m_path_NASA, group_names, raw_nasa_only=False)
    nasa_handler.read_nasa()
    nasa_handler.nasa_average(start=2)  # Calculate average score and add to dataframe
    nasa_handler.plot_sub_data(
        start=2,
        fig_design=(2, 3),
        fig_size=(6, 5),
        subplot_titles=plot_titles,
        same_yaxis=np.arange(0, 101, 20),
        p_correction=False,
    )
    nasa_handler.save_fig()

    howMuchQuesPerGroup = 15
    validQuesStartFrom = 13
    onlyDrawMeanForQues = True
    group_names = ["A", "B", "C"]
    path = os.path.expanduser(os.getenv("GOOGLE_FORM_PATH"))  # Path to the csv

    google_handler = GoogleQuesProcess(path, group_names)
    google_handler.read_clean_column_names(" - ")
    # google_handler.df = google_handler.df.rename(columns={'TR2': 'IT1'})  # Adjust some unintended columns
    print(
        "Mean age:",
        google_handler.df["BC2"].mean(),
        "Std:",
        google_handler.df["BC2"].std(),
    )
    google_handler.plot_bar(
        validQuesStartFrom,
        howMuchQuesPerGroup,
        mean_ques=onlyDrawMeanForQues,
        fig_size=(6, 4),
        draw_mean=True,
        p_correction=False,
    )
    google_handler.save_fig()
