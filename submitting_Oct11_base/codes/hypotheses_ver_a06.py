

import tqdm
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact


def perform_mann_whitney_test(
        label = "label.", 
        x = np.array([6, 7, 8, 10, 20]), 
        y = np.array([0, 0, 1, 1, 2]), 
        alternative = 'greater', 
        # Select alternative='greater' or "less".
        # If "greater", x is greater when significant.
        verbose = False, debug = False):
    """
    Imports: 
    from scipy.stats import mannwhitneyu
    import numpy as np
    """
    result = mannwhitneyu(
            x = x, 
            y = y, 
            alternative = alternative, 
            nan_policy = 'omit',
            method = "asymptotic", 
            use_continuity = True, 
            )
    stat, pvalue = result

    if verbose: 
        print("# # %s, Zeroes in test: x: %d / %d, y: %d / %d" % (
                label, sum(x == 0), len(x), sum(y == 0), len(y), 
                ))
        print(pvalue)
    
    if debug:
        print(result)
        print(type(result))
        print(pvalue)
    return stat, pvalue, len(x), len(y)


def get_priority_frame_for_traits_by_mann_whitney_test(
        resigned_frame, control_frame, 
        traits = [], 
        alternative = 'greater', 
        sanitize_zero = True, 
        verbose = False, debug = False, ):
    records = list()
    for trait in tqdm.tqdm(traits):
        x = resigned_frame[trait]
        y = control_frame[trait]
        if sanitize_zero:
            # Sanitize input.
            x = x[x > 0]; y = y[y > 0]; 

        stat, pvalue, count_of_x, count_of_y = perform_mann_whitney_test(
                label = "Trait: " + trait, 
                x = x, 
                y = y, 
                alternative = alternative, 
                verbose = verbose, 
                )
        records.append(dict(
                trait = trait, pvalue = pvalue, 
                side = alternative, 
                type_of_test = "Mann_Whitney_U", 
                count_of_x = count_of_x, 
                count_of_y = count_of_y, 
                ))
    
    frame = pd.DataFrame(records)
    frame = frame.sort_values("pvalue", ascending = True)
    frame = frame.reset_index(drop = True)
    if debug: print(frame)
    return frame


def select_fourteen_traits(header):
    # Select fourteen personality traits.
    def starts_with(query = "未・", text = ""):
        return text[0:len(query)] == query
    fourteen_traits = [
            text for text in header
            if starts_with("未・", text) or starts_with("既・", text)
            ]
    return fourteen_traits


def save_priority_frame_for_simple_traits_in_resignation(
        in_file_path = "./datasets/case_master_var_fourteen.tsv", 
        out_file_path = "./datasets/greater_mwu_table_for_resignation.tsv", 
        debug = False, **kwargs
        ):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    header = staff_frame.columns.tolist()
    fourteen_traits = select_fourteen_traits(header)
    resigned_frame = select_resigned_frame(staff_frame)
    nonresigned_frame = get_diff_frame(staff_frame, resigned_frame, column = "staff")
    if debug:
        print(header)
        print(fourteen_traits)
        print(f"{len(resigned_frame)=}")
        print(f"{len(nonresigned_frame)=}")

    priority_frame = get_priority_frame_for_traits_by_mann_whitney_test(
            resigned_frame, nonresigned_frame, 
            fourteen_traits, 
            alternative = 'greater', 
            verbose = True, )
    # In this formulation, alternative = 'less' is not very meaningful.
    # It indicates values are closer to 50.
    priority_frame.to_csv(out_file_path, sep = "\t", index = False)
    return priority_frame


def add_binary_trait_values_to_staff_frame(
        trait_a = "未・社交性", 
        trait_b = "既・慎重性", 
        staff_frame = None, 
        mode = "diff", 
        column = "diff", 
        copy = True, 
        ):
    if copy:
        score_frame = staff_frame.copy()
    else:
        score_frame = pd.DataFrame()

    if mode == "sum":
        score_frame[column] = staff_frame[trait_b] + staff_frame[trait_a]
    elif mode == "diff":
        score_frame[column] = staff_frame[trait_b] - staff_frame[trait_a]
    elif mode == "ratio":
        score_frame[column] = (staff_frame[trait_b] + 1.0) / (staff_frame[trait_a] + 1.0)
    else:
        score_frame = pd.DataFrame()
    return score_frame


def get_binary_trait_label(
        trait_a = "未・社交性", 
        trait_b = "既・慎重性", 
        mode = "diff", 
        ):
    if mode == "sum":
        label = "[%s]-plus-[%s]" % (trait_b, trait_a)
    elif mode == "diff":
        label = "[%s]-minus-[%s]" % (trait_b, trait_a)
    elif mode == "ratio":
        label = "[%s]-over-[%s]" % (trait_b, trait_a)
    else:
        label = "[%s]-XXX-[%s]" % (trait_b, trait_a)
    return label


def get_formula_frame_for_trait(
        trait_a = "未・社交性", 
        trait_b = "既・慎重性", 
        staff_frame = None, 
        mode = "diff", 
        ):
    
    score_frame = add_binary_trait_values_to_staff_frame(
            trait_a = trait_a, 
            trait_b = trait_b, 
            staff_frame = staff_frame, 
            mode = mode, 
            column = mode, 
            copy = True,
            )
    label = get_binary_trait_label(
            trait_a = trait_a, 
            trait_b = trait_b, 
            mode = mode, )
    
    header = [mode, "staff", trait_b, trait_a]
    temp_frame = score_frame[header].copy()
    temp_frame["label"] = [label] * len(temp_frame)
    formula_frame = temp_frame[["label"] + header]
    return label, formula_frame


def get_formula_frame_for_select_traits(
        mode = "diff", 
        traits = [], 
        staff_frame = None, 
        debug = False, ):
    
    lex_of_staffs = dict()
    for trait_a in traits:
        for trait_b in traits:
            if trait_a != trait_b:
                label, formula_frame = get_formula_frame_for_trait(
                        trait_a = trait_a, 
                        trait_b = trait_b, 
                        staff_frame = staff_frame, 
                        mode = mode, 
                        )
                lex_of_staffs[label] = formula_frame
                if debug:
                    print(score_frame[["staff", trait_a, trait_b]])
    submit_frame = pd.concat(list(lex_of_staffs.values()), ignore_index = True)
    submit_frame = submit_frame.sort_values(mode, ascending = False)
    submit_frame = submit_frame.reset_index(drop = True)
    return submit_frame


def save_personal_trait_formula_frame_for_fourteen_traits(
        mode = "diff", 
        in_file_path = "./datasets/case_resigned_var_fourteen.tsv", 
        out_file_path = "./datasets/case_resigned_var_fourteen_explore_diff.tsv", 
        debug = False, **kwargs):
    """
    # Analyze combinations of the fourteen trait variables.
    # Define sums, differences [字・活動性]-minus-[字・社交性] and ratios [字・活動性]-over-[字・社交性].
    """
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    fourteen_traits = select_fourteen_traits(staff_frame.columns)
    submit_frame = get_formula_frame_for_select_traits(
            mode = mode, 
            traits = fourteen_traits, 
            staff_frame = staff_frame, 
            )
    submit_frame.to_csv(out_file_path, sep = "\t", index = False)
    return submit_frame


def get_trait_frame_with_binary_trait_columns(
            mode = "diff", 
            staff_frame = None, traits = [], 
            verbose = False, ):
    # Make sum, diff, and ratio variables.
    binary_frame = pd.DataFrame()
    for trait_a in traits:
        for trait_b in traits:
            if trait_a != trait_b:
                label = get_binary_trait_label(
                        trait_a = trait_a, 
                        trait_b = trait_b, 
                        mode = mode, )
                if verbose: print(label)

                score_frame = add_binary_trait_values_to_staff_frame(
                        trait_a = trait_a, 
                        trait_b = trait_b, 
                        staff_frame = staff_frame, 
                        mode = mode, 
                        column = label, 
                        copy = False, 
                        )
                binary_frame = pd.concat([binary_frame, score_frame], axis = 1)
    return binary_frame


def get_binary_trait_frame(
            staff_frame, 
            modes = ["diff", "ratio", "sum"], 
            traits = [], 
            debug = False, ):
    header = staff_frame.columns.tolist()
    frame = staff_frame.copy()
    for mode in modes:
        binary_frame = get_trait_frame_with_binary_trait_columns(
                mode = mode, 
                staff_frame = staff_frame, 
                traits = traits, 
                verbose = False, )
        if debug:
            print(binary_frame)
        frame = pd.concat([frame, binary_frame], axis = 1)
    return frame


def save_trait_frame_with_all_binary_trait_columns(
            in_file_path = "./datasets/case_master_var_fourteen.tsv", 
            out_file_path = "./datasets/case_master_var_fourteen_with_binary_traits.tsv", 
            modes = ["diff", "ratio", "sum"], 
            debug = False, **kwargs):
            
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    fourteen_traits = select_fourteen_traits(staff_frame.columns)
    frame = get_binary_trait_frame(
            staff_frame, 
            modes = modes, 
            traits = fourteen_traits, 
            debug = debug, )
    frame.to_csv(out_file_path, sep = "\t", index = False)
    return frame


def save_binary_trait_frame_in_seven_traits(
            in_file_path = "./datasets/case_master_var_fourteen.tsv", 
            out_file_path = "./datasets/case_master_var_seven_with_binary_traits.tsv", 
            modes = ["diff", "ratio", "sum"], 
            debug = False, **kwargs):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    traits = staff_frame.columns[6:13]
    frame = get_binary_trait_frame(
            staff_frame, 
            modes = modes, 
            traits = traits, 
            debug = debug, )
    frame.to_csv(out_file_path, sep = "\t", index = False)
    return frame


def get_binary_trait_columns(columns):
    atoms = list()
    for column in columns:
        str_list = column.split("-")
        if len(str_list) > 1:
            if str_list[1] in ["over", "minus", "plus"]:
                atoms.append(column)
    return atoms


def save_priority_frame_by_mann_whitney_for_binary_traits(
        in_file_path = "./datasets/case_master_var_fourteen_with_binary_traits.tsv", 
        out_file_path = "./datasets/greater_mwu_table_for_resignation_with_binary_traits.tsv", 
        alternative = 'greater', 
        debug = False, **kwargs
        ):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    header = staff_frame.columns.tolist()
    resigned_frame = select_resigned_frame(staff_frame)
    nonresigned_frame = get_diff_frame(staff_frame, resigned_frame, column = "staff")
    
    binaries = get_binary_trait_columns(staff_frame.columns)
    if debug:
        print(resigned_frame)
        print(binaries)

    priority_frame = get_priority_frame_for_traits_by_mann_whitney_test(
            resigned_frame, nonresigned_frame, 
            traits = binaries, 
            alternative = alternative, 
            verbose = False, )
    priority_frame.to_csv(out_file_path, sep = "\t", index = False)
    return priority_frame



