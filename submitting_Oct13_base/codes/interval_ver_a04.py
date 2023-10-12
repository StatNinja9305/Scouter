
import tqdm
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Custom imports.
from toolbox import Staff_Table


def get_intervals_for_trait_of_seven(
        resigned_frame, 
        master_frame, 
        trait = "慎重性", 
        alpha = 0.3, 
        debug = False, ):
    values_total = master_frame[trait].tolist()
    values_resigned = resigned_frame[trait].tolist()

    def get_occurrences(values):
        occurs = np.zeros((100))
        for i in range(len(values)):
            occurs[int(values[i] - 1.0)] += 1.0
        return occurs
    
    occurs_total = get_occurrences(values_total)
    occurs_resigned = get_occurrences(values_resigned)

    def get_interval_rate(
            occurs_resigned, 
            occurs_total, 
            lower = 40, 
            upper = 60, 
            ):
        def get_count(occurs):
            # Globals: lower. upper
            return np.sum(occurs[int(lower - 1):int(upper - 1)])
        count_of_total = get_count(occurs_total)
        count_of_resigned = get_count(occurs_resigned)
        rate = count_of_resigned / count_of_total
        report = dict(
                rate = rate, 
                count_of_total = count_of_total, 
                count_of_resigned = count_of_resigned, 
                lower = lower, 
                upper = upper, 
                )
        return report
    
    # Get intervals.
    reports = list()
    for lower in tqdm.tqdm(range(1, 101, 1)):
        for upper in range(lower + 1, 101, 1):
            report = get_interval_rate(
                    occurs_resigned, 
                    occurs_total, 
                    lower = lower, 
                    upper = upper, 
                    )
            if report["rate"] >= alpha:
                reports.append(report)
    if debug:
        print(values_total)
        print(occurs_total)
    return reports


def get_mesh_for_occurences(values):
    """ Returns a mesh with 100 bins.
    """
    values = np.array(values)
    my_max = np.max(values)
    my_min = np.min(values)

    step = (my_max - my_min) / (100 - 1)
    meshes = np.array([my_min + step * i for i in range(100 + 1)])
    return meshes


def get_occurrences_by_mesh(values):
    """ Returns frequencies in 100 bins in mesh.
    """
    values = np.array(values)
    my_max = np.max(values)
    my_min = np.min(values)

    meshes = get_mesh_for_occurences(values)
    occurs = np.zeros((100))
    for value in values:
        indices = np.where(meshes > value)[0]
        index = indices[0] - 1
        occurs[index] += 1.0
    return dict(
            occurs = occurs, 
            meshes = meshes, 
            my_max = my_max, 
            my_min = my_min, 
            )


def get_intervals_for_binary_traits(
        resigned_frame, 
        master_frame, 
        trait = '[活動性]-over-[固執性]', 
        alpha = 0.5, 
        debug = False, ):

    values_total = np.array(master_frame[trait].tolist())
    values_resigned = np.array(resigned_frame[trait].tolist())
    board_of_total = get_occurrences_by_mesh(values_total)
    board_of_resigned = get_occurrences_by_mesh(values_resigned)

    def get_count_with_mesh(
            occurs, meshes, 
            lower = 2.0, 
            upper = 2.5, 
            debug = False, 
            ):
        # Count occurrences.
        indices = np.where((lower < meshes) & (meshes < upper))[0]
        indices = indices[indices < 100]
        count = np.sum([occurs[i] for i in indices])
        if debug: 
            print(indices)
            print([meshes[i] for i in indices])
            print(count)
            print(len(occurs), len(meshes))
            frame = pd.DataFrame(dict(meshes = meshes, occurs = list(occurs) + [0]))
            [print(row) for index, row in frame.iterrows()]
        return count
    
    def get_interval_rate_with_mesh(
            board_of_total, 
            board_of_resigned, 
            lower = 2.0, 
            upper = 2.5, 
            ):
        count_of_total = get_count_with_mesh(
                occurs = board_of_total["occurs"], 
                meshes = board_of_total["meshes"], 
                lower = lower, 
                upper = upper, 
                )
        count_of_resigned = get_count_with_mesh(
                occurs = board_of_resigned["occurs"], 
                meshes = board_of_resigned["meshes"], 
                lower = lower, 
                upper = upper, 
                )
        if count_of_total > 0:
            rate = count_of_resigned / count_of_total
        else:
            rate = np.nan
        report = dict(
                rate = rate, 
                count_of_total = count_of_total, 
                count_of_resigned = count_of_resigned, 
                lower = lower, 
                upper = upper, 
                )
        return report
    
    # Get intervals.
    reports = list()
    meshes = get_mesh_for_occurences(values_total)
    for lower in tqdm.tqdm(meshes):
        select_meshes = meshes[meshes > lower]
        #print(len(select_meshes))
        for upper in select_meshes:
            report = get_interval_rate_with_mesh(
                    board_of_total, 
                    board_of_resigned, 
                    lower = lower, 
                    upper = upper, 
                    )
            rate = report["rate"]
            if rate is not np.nan:
                if rate >= alpha:
                    reports.append(report)
    if debug:
        print(values_total)
        print(reports[:20])
        print(len(reports))
    return reports


def save_table_of_intervals(
        in_file_path = "./datasets/case_master.tsv", 
        out_file_path = "./datasets/intervals_of_seven.tsv", 
        mode = "seven", 
        debug = False, 
        **kwargs):
    master = Staff_Table(in_file_path)
    resigned_frame = master.get_resigned_frame()

    if mode == "seven":
        traits = master.get_seven_traits()
        get_intervals_by_custom_method = get_intervals_for_trait_of_seven
    elif mode == "binary":
        traits = master.get_binary_traits()
        get_intervals_by_custom_method = get_intervals_for_binary_traits
    else:
        traits = master.get_seven_traits()
        get_intervals_by_custom_method = get_intervals_for_trait_of_seven
    
    if debug:
        print(traits)
        print(resigned_frame)

    submit_frame = pd.DataFrame()
    for trait in traits:
        print(trait)
        reports = get_intervals_by_custom_method(
                resigned_frame, 
                master.frame, 
                trait = trait, 
                alpha = 0.5, 
                )
        
        if len(reports) > 0:
            # Format table.
            head_frame = pd.DataFrame(dict(trait = [trait] * len(reports)))
            report_frame = pd.DataFrame(reports)
            frame = pd.concat([head_frame, report_frame], axis = 1)
            frame["width"] = frame["upper"] - frame["lower"]
            frame = frame.sort_values("width", ascending = False)
            frame = frame.reset_index(drop = True)
            submit_frame = pd.concat([submit_frame, frame], axis = 0)
        if debug:
            print(len(reports))
            print(reports[:3])
    submit_frame = submit_frame.reset_index(drop = True)
    submit_frame.to_csv(out_file_path, sep = "\t", index = False)
    return submit_frame


def add_intervals_by_frame(ax, trait_frame, i = 5):
    for index, row in trait_frame.iterrows():
        #print(row["lower"], row["upper"])
        width = row["upper"] - row["lower"]
        height_half = 0.1
        y_lower = float(i) - height_half
        height = 2 * height_half
        rectangle = patches.Rectangle(
                # 長方形を作成 (左下のx座標, 左下のy座標, 幅, 高さ)
                (row["lower"], y_lower), width, height, 
                edgecolor='red', facecolor='red', 
                alpha = 0.3, 
                )
        # 長方形をプロットエリアに追加
        ax.add_patch(rectangle)
    return ax


def save_plot_of_intervals_of_basic_traits(
        in_file_path = "./datasets/intervals_of_seven.tsv", 
        out_file_path = "./datasets/intervals_of_seven.pdf", 
        show = False, 
        **kwargs):
    interval_frame = pd.read_csv(in_file_path, sep = "\t")
    traits = sorted(list(set(interval_frame["trait"].tolist())))
    
    print(interval_frame)
    print(traits)

    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'IPAexGothic'

    # 新しい図を作成 size: (width, height)
    fig, ax = plt.subplots(figsize = (4, 4))
    
    for i in range(len(traits)):
        trait = traits[i]
        trait_frame = interval_frame[interval_frame["trait"] == trait]
        #print(trait_frame)
        ax = add_intervals_by_frame(ax, trait_frame, i = i)

    # 座標軸の表示範囲を設定
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 4.5)

    # 複数の縦線を追加
    for i in range(0, 100, 5):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)
    
    # 複数の横線を追加
    for j in range(0, 5, 1):
        plt.axhline(y = j, color = 'black', linestyle = '-', linewidth = 1.0)

    # 左側の余白を広げる
    plt.subplots_adjust(left = 0.2)  # 値を大きくすると、左側の余白が広がる

    # y軸の数値ラベルをカスタム文字に変更
    plt.yticks(range(len(traits)), traits)
    
    plt.title(f"早期離職の50%危険区間：　基本性格変数")
    # 軸ラベルの設定
    plt.xlabel("性格変数の数値")
    plt.ylabel("性格変数")
    
    # 図をPDFとして保存
    plt.savefig(out_file_path)
    if show: plt.show()
    return 


def save_plot_of_intervals_of_binary_traits(
        in_file_path = "./datasets/intervals_of_binary.tsv", 
        out_file_path = "./datasets/intervals_of_binary_of_plus.pdf", 
        mode = "plus", 
        show = False, 
        alpha = 1.0, 
        top = 50, 
        debug = False, 
        **kwargs):
    
    operator = mode
    interval_frame = pd.read_csv(in_file_path, sep = "\t")
    traits = sorted(list(set(interval_frame["trait"].tolist())))
    
    def get_traits_of_operator(traits, operator = "plus"):
        atoms = list()
        for trait in traits:
            if trait.split("-")[1] == operator:
                atoms.append(trait)
        return atoms
    
    def select_trait_by_operator(interval_frame, operator = "plus"):
        traits = sorted(list(set(interval_frame["trait"].tolist())))
        traits_select = get_traits_of_operator(traits, operator)
        select_frame = interval_frame[
                interval_frame['trait'].isin(traits_select)].copy()
        select_frame = select_frame.sort_values("width", ascending = False)
        select_frame = select_frame.reset_index(drop = True)
        return select_frame

    case_frame = select_trait_by_operator(interval_frame, operator = operator)
    case_frame = case_frame[case_frame["rate"] >= alpha]
    if debug:
        print(interval_frame)
        print(len(traits))
        print(case_frame)

    def sanitize_pluses(traits):
        traits = list(set(traits))
        atoms = list()
        for trait in traits:
            strs = trait.split("-")
            inverse = strs[2] + "-" + strs[1] + "-" + strs[0]
            if inverse not in atoms:
                atoms.append(trait)
        return atoms
    
    traits_raw = list(set(case_frame["trait"].tolist()))
    if operator == "plus":
        # Cancel equal ones as definition (A + B = B + A).
        traits_raw = sanitize_pluses(traits_raw)
        
    print("# of traits:", len(traits_raw))

    # Truncate tops.
    traits = traits_raw[:top]
    traits = sorted(traits)

    # Reverse list for visualization.
    traits = list(reversed(traits))
    #print(traits)
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'IPAexGothic'

    # 新しい図を作成 size: (width, height)
    fig, ax = plt.subplots(figsize = (4, 8))

    # Add intervals.
    for i in range(len(traits)):
        trait = traits[i]
        trait_frame = case_frame[case_frame["trait"] == trait]
        ax = add_intervals_by_frame(ax, trait_frame, i = i)

    # 座標軸の表示範囲を設定
    my_min = min(case_frame["lower"].tolist())
    my_max = max(case_frame["upper"].tolist())
    width = abs(my_max - my_min)
    offset = width / 20
    ax.set_xlim(my_min - offset, my_max + offset)
    ax.set_ylim(- 0.5, len(traits) - 0.5)

    # 複数の縦線を追加
    if operator in ["plus", "minus"]:
        unit = 5
    elif operator == "over":
        unit = 0.1
    else: unit = 5
    i_start = int(my_min / unit) - 1
    i_end = int(my_max / unit) + 1
    verticals = [unit * i for i in range(i_start, i_end)]
    for x in verticals:
        plt.axvline(x = x, color='gray', linestyle='--', linewidth=0.5)
    
    # 複数の横線を追加
    for j in range(len(traits)):
        plt.axhline(y = j, color = 'black', linestyle = '-', linewidth = 1.0)

    # 左側の余白を広げる
    plt.subplots_adjust(left = 0.45)  # 値を大きくすると、左側の余白が広がる
    plt.subplots_adjust(top = 0.93)
    plt.subplots_adjust(bottom = 0.08)

    # y軸の数値ラベルをカスタム文字に変更
    plt.yticks(range(len(traits)), traits)
    
    plt.title(f"早期離職の{int(alpha * 100)}%危険区間：\n{operator}-性格変数")
    # 軸ラベルの設定
    plt.xlabel("性格変数の数値")
    plt.ylabel("性格変数")
    
    # 図をPDFとして保存
    plt.savefig(out_file_path)
    if show: plt.show()

    return
