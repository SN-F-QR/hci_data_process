import os
import numpy as np
import log_data as handler


if __name__ == "__main__":
    group_names = ["A", "B", "C", "D"]  # Each Group Name
    plot_titles = ['time', 'behavior1', 'behavior2', 'behavior3', 'behavior4', 'behavior5']  # Titles for each subplot
    path = os.path.expanduser("~/Developer/Exp_Result/finished")  # Path to folder including data files
    # Draw and Analyze Unity Json Data
    unity_handler = handler.UnityJsonProcess(path, group_names)
    unity_handler.read_jsons(where_id=1, where_group=2)
    unity_handler.plot_sub_data(start=2, fig_design=(1, 6), fig_size=(12, 3), subplot_titles=plot_titles,
                                group_colors=None, p_corretion=False)
    unity_handler.save_fig()  # Always to save Figure after all adjustments has conducted

    # Draw and Analyze NASA TLX data
    m_path_NASA = os.path.expanduser("~/Developer/Exp_Result/NASA_TLX")  # Path to folder including data files
    plot_titles = ['mental', 'physical', 'temporal', 'performance', 'effort', 'frustration']
    nasa_handler = handler.TLXProcess(m_path_NASA, ["A", "B", "C", "D"], raw_nasa=False)
    nasa_handler.read_nasa()
    nasa_handler.nasa_average(start=2)  # Calculate average score and add to dataframe
    nasa_handler.plot_sub_data(start=2, fig_design=(2, 3), fig_size=(6, 5), subplot_titles=plot_titles,
                               group_colors=None, same_yaxis=np.arange(0, 101, 20), p_corretion=False)
    nasa_handler.save_fig()  # Always to save Figure after all adjustments has conducted



    # csv_frame = pd.DataFrame({"number": number, "point_0": point[0], "point_1": point[1], "point_2": point[2]})
    # csv_frame.to_csv(os.path.join(m_path, "csv\sum.csv"), index=False)
