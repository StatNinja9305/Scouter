




from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from toolbox import (
        select_resigned_frame, 
        get_trait_frame, 
        get_advanced_frame, 
        )
from hypotheses import (
        get_binary_trait_columns, 
        get_priority_frame_for_traits_by_mann_whitney_test, 
        )


def get_upper_frame_by_threshold(
        trait_frame, 
        column = "慎重性", 
        top_percent = 10, 
        # Equal or greater than this threshold.
        ):
    values_of_total = trait_frame[column].tolist()
    # A threshold by lower percentile.
    threshold = np.percentile(values_of_total, 100 - top_percent)
    upper_frame = trait_frame[trait_frame[column] >= threshold]
    return upper_frame



def get_contingency_table_of_resigned_by_threshold(
        upper_frame, trait_frame, 
        ):
    count_of_upper_resigned = len(select_resigned_frame(upper_frame))
    count_of_total_resigned = len(select_resigned_frame(trait_frame))
    contingency_table = pd.DataFrame([
            [count_of_upper_resigned, count_of_total_resigned], 
            [len(upper_frame), len(trait_frame)], 
            ])
    return contingency_table


def get_pvalue_of_fisher_exact_by_threshold(
        trait_frame, 
        top_percent = 10, 
        column = "慎重性", 
        debug = False):
    upper_frame = get_upper_frame_by_threshold(
            trait_frame, 
            column = column, 
            top_percent = top_percent, 
            )
    contingency_table = get_contingency_table_of_resigned_by_threshold(
            upper_frame, trait_frame, 
            )
    stat, pvalue = fisher_exact(contingency_table)
    if debug:
        print(contingency_table)
        print(stat, pvalue)
    return pvalue


def save_pvalue_table_of_fisher_exact_in_resigned(
        in_file_path = "./datasets/case_master.tsv", 
        out_file_path = "./datasets/fisher_exact_pvalue_table_for_resignation.tsv", 
        **kwargs):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    trait_frame = get_trait_frame(staff_frame)
    
    thetas = range(5, 100, 5)
    pvalues = [
            get_pvalue_of_fisher_exact_by_threshold(
                    trait_frame, theta, 
                    column = "慎重性", 
                    ) 
            for theta in thetas]
    frame = pd.DataFrame({
            "top_careful_percent": thetas, 
            "pvalue": pvalues, 
            "test": ["Fisher_Exact"] * len(thetas), 
            })
    frame.to_csv(out_file_path, sep = "\t", index = False)
    return frame



def save_contingency_table_of_resigned(
        in_file_path = "./datasets/case_master.tsv", 
        out_file_path = "./datasets/fisher_exact_contingency_table_for_resignation.tsv", 
        theta = 10, 
        debug = False, **kwargs):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    trait_frame = get_trait_frame(staff_frame)
    upper_frame = get_upper_frame_by_threshold(
            trait_frame, 
            column = "慎重性", 
            top_percent = theta, 
            )
    contingency_table = get_contingency_table_of_resigned_by_threshold(
            upper_frame, trait_frame, 
            )
    contingency_table.columns = [
            "EAbove_Upper" + str(theta), 
            "Total" + str(theta)]
    contingency_table.index = ["Resigned", "Total"]
    if debug:
        print(contingency_table)
        [print(row[0], row[1]) for i, row in contingency_table.iterrows()]
        #print(contingency_table.apply(, axis = 0))
        ratios = contingency_table.iloc[0, :] / contingency_table.iloc[1, :]
        print(ratios)
    contingency_table.to_csv(out_file_path, sep = "\t", index = True)
    return contingency_table


def save_upper_frame_by_threshold_against_carefulness(
        in_file_path = "./datasets/case_master_with_binary_traits_out_of_seven.tsv", 
        out_file_path = "./datasets/upper_carefulness_by_theta_25.tsv", 
        theta = 25, 
        **kwargs):
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    upper_frame = get_upper_frame_by_threshold(
            staff_frame, 
            column = "慎重性", 
            top_percent = theta, 
            )
    upper_frame.to_csv(out_file_path, sep = "\t", index = False)



def select_seven_traits(staff_frame):
    traits = staff_frame.columns[6:13]
    return traits


def save_mann_whitney_priority_frame_of_traits(
        in_file_path = "./datasets/upper_carefulness_by_theta_25.tsv", 
        out_file_path = "./datasets/greater_mwu_table_for_carefulness_with_theta_25_with_seven_traits.tsv", 
        alternative = "greater", 
        mode = "seven", 
        debug = False, **kwargs):
    upper_frame = pd.read_csv(in_file_path, sep = "\t")
    upper_resigned_frame = select_resigned_frame(upper_frame)
    upper_advanced_frame = get_advanced_frame(upper_frame)
    
    # Switch traits between the seven traits and their binary traits.
    if mode == "seven":
        traits = select_seven_traits(upper_resigned_frame)
    elif mode == "binary":
        traits = get_binary_trait_columns(upper_resigned_frame)
    else:
        traits = select_seven_traits(upper_resigned_frame)
    
    # Comparison of two groups.
    # Upper resigned versus upper advanced.
    priority_frame = get_priority_frame_for_traits_by_mann_whitney_test(
            upper_advanced_frame, 
            upper_resigned_frame, 
            traits = traits, 
            alternative = alternative, 
            sanitize_zero = False, 
            verbose = False, )
    if debug:
        print(len(upper_resigned_frame), len(upper_advanced_frame))
        print(traits)
        print(priority_frame)
    priority_frame.to_csv(out_file_path, sep = "\t", index = False)
    return priority_frame





def visualize_fisher_exact_pvalue_table_for_resignation(
        in_file_path = "./datasets/fisher_exact_pvalue_table_for_resignation.tsv", 
        out_file_path = "./datasets/fisher_exact_pvalue_table_for_resignation.pdf", 
        show = False, **kwargs):
    score_frame = pd.read_csv(in_file_path, sep = "\t")
    thetas = score_frame["top_careful_percent"]
    pvalues = score_frame["pvalue"]

    fig = plt.figure(facecolor="w")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(thetas, pvalues, color="b")
    ax.text(-10, 0.20, "P = 0.20")
    ax.text(-10, 0.10, "P = 0.15")
    ax.text(-10, 0.05, "P = 0.05")

    plt.yscale("log")
    plt.xlabel("Threshold for Carefulness (Top %)")
    plt.ylabel("P-value")
    plt.title("Fisher Exact P-values",fontsize = 20)
    plt.hlines(0.20, -10, 110, color = "black", linewidth = 0.5)
    plt.hlines(0.10, -10, 110, color = "black", linewidth = 0.5)
    plt.hlines(0.05, -10, 110, color = "black", linewidth = 0.5)

    plt.vlines(25, 0.0, 1.0, color = "red", linewidth = 0.5)
    plt.vlines(10, 0.0, 1.0, color = "red", linewidth = 0.5)

    plt.grid()
    plt.savefig(out_file_path)
    if show:
        plt.show()
        print(score_frame)
    return 
