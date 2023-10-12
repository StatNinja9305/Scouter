
import numpy as np
import pandas as pd
import json
import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from contingency import (
        select_seven_traits,
        )
from toolbox import (
        get_trait_frame, 
        select_resigned_frame, 
        )


def show_frequency_matrix(matrix, axis = 0):
    results = np.apply_along_axis(
            np.sum, axis = axis, arr = matrix)
    # 1-origin indexing.
    show = list(zip(range(1, len(results) + 1), results))
    print(json.dumps(show, indent = 4, ensure_ascii = False))


def get_frequency_matrix_of_two_traits(
        staff_frame, 
        trait_x = "慎重性", 
        trait_y = "社交性", 
        normalize = True, 
        debug = False):
    values_x = staff_frame[trait_x].tolist()
    values_y = staff_frame[trait_y].tolist()

    occurrence = np.zeros((100, 100))
    for i in range(len(values_x)):
        x = values_x[i]
        y = values_y[i]
        # x and y are actual values.
        # x in columns (x-axis) and y in rows (y-axis).
        occurrence[int(y - 1.0), int(x - 1.0)] += 1.0
    if normalize:
        occurrence /= len(values_x)
    if debug:
        show_frequency_matrix(occurrence)
    return occurrence


def get_rectangle_rate(
        occur_resigned, occur_total, 
        y_lower = 65, 
        y_upper = 70, 
        x_lower = 48, 
        x_upper = 50, 
        ):
    # Define a rectangle range.
    def get_rectangle(matrix):
        return matrix[
                int(y_lower - 1):int(y_upper - 1), 
                int(x_lower - 1):int(x_upper - 1), ]
    # Values >= lower and < upper
    count_of_total = np.sum(get_rectangle(occur_total))
    count_of_resigned = np.sum(get_rectangle(occur_resigned))
    rate = count_of_resigned / count_of_total
    return dict(
            rate = rate, 
            count_of_total = count_of_total, 
            count_of_resigned = count_of_resigned, 
            y_lower = y_lower, 
            y_upper = y_upper, 
            x_lower = x_lower, 
            x_upper = x_upper, 
            )


def get_rectangles_for_50_percent_risk_zone_for_trait(
        staff_frame, 
        trait_x = "慎重性", 
        trait_y = "社交性", 
        debug = False, 
        ):
    """
    Functions called:
            select_seven_traits, 
            get_trait_frame, 
            select_resigned_frame, 
            get_frequency_matrix_of_two_traits, 
    """
    trait_frame = get_trait_frame(staff_frame)
    resigned_frame = select_resigned_frame(staff_frame)
    if debug:
        print(trait_frame.columns)
        print(len(resigned_frame))

    # Get a probability matrix.
    # Assume 1...100 integer values.
    occur_total = get_frequency_matrix_of_two_traits(
            staff_frame, 
            trait_x = trait_x, 
            trait_y = trait_y, 
            normalize = False, 
            )
    occur_resigned = get_frequency_matrix_of_two_traits(
            resigned_frame, 
            trait_x = trait_x, 
            trait_y = trait_y, 
            normalize = False, 
            )
    
    def get_element(row):
        return [row[0], row[1]]
    pairs = np.apply_along_axis(
            get_element, axis = 1, arr = resigned_frame[[trait_x, trait_y]])
    #print(pairs)

    # Restrict range of search to observation plus/minus 10.
    tuples = list()
    for pair in tqdm.tqdm(pairs):
        x = pair[0]
        y = pair[1]
        for y_lower in range(y - 10, y + 11):
            for y_upper in range(y_lower + 1, y + 11):
                for x_lower in range(x - 10, x + 11):
                    for x_upper in range(x_lower + 1, x + 11):
                        tuples.append((y_lower, y_upper, x_lower, x_upper))
    
    # Get rectangles.
    reports = list()
    for params in tqdm.tqdm(tuples):
        report = get_rectangle_rate(
                occur_resigned, occur_total, 
                y_lower = params[0], 
                y_upper = params[1], 
                x_lower = params[2], 
                x_upper = params[3], 
                )
        if report["rate"] >= 0.5:
            reports.append(report)
    if debug:
        print(json.dumps(reports[:40], indent = 4, ensure_ascii = False))
        print(len(reports))
    # Sanitize intervals.
    frame = pd.DataFrame(reports)
    frame = frame.sort_values("rate", ascending = False)
    frame = frame.reset_index(drop = True)
    
    # Add labels for intervals.
    frame["x_label"] = [
            str(int(row["x_lower"])) + ":" + str(int(row["x_upper"])) 
            for index, row in frame.iterrows()
            ]
    frame["y_label"] = [
            str(int(row["y_lower"])) + ":" + str(int(row["y_upper"])) 
            for index, row in frame.iterrows()
            ]
    frame["x_width"] = frame["x_upper"] - frame["x_lower"]
    frame["y_width"] = frame["y_upper"] - frame["y_lower"]
    #print(frame[:20])

    def get_best_record(
            frame, label = "64:75", 
            width = "y_width", 
            column= "x_label", 
            ):
        # Select only one record that has the greatest width.
        temp_frame = frame[frame[column] == label]
        temp_frame = temp_frame.sort_values(width, ascending = False)
        temp_frame = temp_frame.reset_index(drop = True)
        return temp_frame.iloc[0]
    
    def get_best_x_records(frame):
        x_labels = sorted(list(set(frame["x_label"].tolist())))
        records = list()
        for label in x_labels:
            record = get_best_record(
                    frame, label = label, 
                    width = "y_width", 
                    column= "x_label", 
                    )
            records.append(record)
        return records
    def get_best_y_records(frame):
        y_labels = sorted(list(set(frame["y_label"].tolist())))
        records = list()
        for label in y_labels:
            record = get_best_record(
                    frame, label = label, 
                    width = "x_width", 
                    column= "y_label", 
                    )
            records.append(record)
        return records
    
    records = get_best_x_records(frame)
    select_x_frame = pd.DataFrame(records)
    records = get_best_y_records(select_x_frame)
    select_frame = pd.DataFrame(records)
    select_frame = select_frame.reset_index(drop = True)
    if debug:
        print(select_frame.loc[:20, ["x_label", "y_label", "x_width", "y_width"]])
        print(len(select_frame))
    return select_frame


def save_rectangles_for_50_percent_risk_zone(
        in_file_path = "./datasets/case_master.tsv", 
        out_file_path = "./datasets/rectangles.tsv", 
        verbose = True, 
        **kwargs):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    traits = list(select_seven_traits(staff_frame))
    trait_x = "慎重性"
    traits.remove(trait_x)
    if verbose: print(traits)

    submit_frame = pd.DataFrame()
    for trait_y in traits:
        if verbose: print(trait_y)
        select_frame = get_rectangles_for_50_percent_risk_zone_for_trait(
                staff_frame, 
                trait_x = trait_x, 
                trait_y = trait_y, 
                debug = False, 
                )
        temp_frame = pd.DataFrame({
                "trait_y": [trait_y] * len(select_frame), 
                "trait_x": [trait_x] * len(select_frame), 
                })
        temp_frame = pd.concat([temp_frame, select_frame], axis = 1)
        submit_frame = pd.concat([submit_frame, temp_frame], axis = 0)
    submit_frame.to_csv(out_file_path, sep = "\t", index = False)
    return submit_frame


def plot_rectangles(
        rect_frame, 
        trait_x = "慎重性", 
        trait_y = "社交性", 
        out_file_path = 'sample_plot.pdf', 
        show = False, 
        debug = False, 
        ):
    trait_frame = rect_frame[
            (rect_frame["trait_y"] == trait_y)
            & (rect_frame["trait_x"] == trait_x)
            ]
    if debug:
        print(trait_frame)
        print(len(trait_frame))

    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'IPAexGothic'

    # 新しい図を作成 size: (width, height)
    fig, ax = plt.subplots(figsize = (5, 5))

    def add_rectangle(
            ax, 
            x_lower = 0.1, 
            x_upper = 0.4, 
            y_lower = 0.1, 
            y_upper = 0.7, 
            ):
        width = x_upper - x_lower
        height = y_upper - y_lower
        rectangle = patches.Rectangle(
                # 長方形を作成 (左下のx座標, 左下のy座標, 幅, 高さ)
                (x_lower, y_lower), width, height, 
                edgecolor='red', facecolor='red', 
                alpha = 0.3, 
                )
        # 長方形をプロットエリアに追加
        ax.add_patch(rectangle)
        return 
    
    for index, row in trait_frame.iterrows():
        #print(row["x_lower"], row["y_upper"])
        add_rectangle(
                ax, 
                x_lower = row["x_lower"], 
                x_upper = row["x_upper"], 
                y_lower = row["y_lower"], 
                y_upper = row["y_upper"], 
                )

    # 座標軸の表示範囲を設定
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 複数の縦線を追加
    for i in range(0, 100, 5):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
    # 複数の横線を追加
    for j in range(0, 100, 5):
        plt.axhline(y=j, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=50, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=50, color='black', linestyle='--', linewidth=0.5)

    plt.title(f"早期離職の50%危険域： X({trait_x}) vs. Y({trait_y})")
    # 軸ラベルの設定
    plt.xlabel(trait_x)
    plt.ylabel(trait_y)
    
    # 図をPDFとして保存
    plt.savefig(out_file_path)
    if show: plt.show()
    
    return


def save_plot_rectangles(
        in_file_path = "./datasets/rectangles.tsv", 
        out_dir_path = "./datasets/rectangles/", 
        **kwargs):
    os.makedirs(out_dir_path, exist_ok=True)
    rect_frame = pd.read_csv(in_file_path, sep = "\t")
    trait_x = list(set(rect_frame["trait_x"].tolist()))[0]
    traits = list(set(rect_frame["trait_y"].tolist()))
    for trait_y in traits:
        print(trait_y)
        plot_rectangles(
                rect_frame, 
                trait_x = trait_x, 
                trait_y = trait_y, 
                out_file_path = f'{out_dir_path}/rectangles_{trait_y}.pdf', 
                show = False, 
                debug = False, 
                )
