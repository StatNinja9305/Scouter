"""


# Env:
conda create -n P9A --clone Python3.9.1
conda activate P9A

# Installation:
pip install -U pip
pip install pandas numpy matplotlib scipy



# Command:

conda activate P9A

ver=a25


python ./look_into_data_ver_${ver}.py \
    --function make_master_table_by_adding_staff_ids \
    --in_file_path "./datasets/table_for_analysis.csv" \
    --out_file_path "./datasets/case_master.tsv" \

python ./look_into_data_ver_${ver}.py \
    --function divide_table_into_resigned_and_nonresigned \
    --in_file_path "./datasets/case_master.tsv" \
    --alpha_file_path "./datasets/case_resigned.tsv" \
    --beta_file_path "./datasets/case_nonresigned.tsv" \

python ./look_into_data_ver_${ver}.py \
    --function divide_table_into_developing_and_nondeveloping \
    --in_file_path "./datasets/case_master.tsv" \
    --alpha_file_path "./datasets/case_developing.tsv" \
    --beta_file_path "./datasets/case_nondeveloping.tsv" \


python ./look_into_data_ver_${ver}.py \
    --function save_personal_trait_formula_frame_for_seven_traits \
    --mode "diff" \
    --alpha_file_path "./datasets/case_resigned.tsv" \
    --out_file_path "./datasets/case_resigned_explore_diff_sort_by_score.tsv" \
    --sort_by_value 1 \


function try_formula_mode () {
    mode=${1}
    python ./look_into_data_ver_${ver}.py \
        --function save_personal_trait_formula_frame_for_seven_traits \
        --mode ${mode} \
        --alpha_file_path "./datasets/case_resigned.tsv" \
        --out_file_path "./datasets/case_resigned_explore_${mode}_sort_by_score.tsv" \
        --sort_by_value 1 \

    python ./look_into_data_ver_${ver}.py \
        --function save_personal_trait_formula_frame_for_seven_traits \
        --mode ${mode} \
        --alpha_file_path "./datasets/case_resigned.tsv" \
        --out_file_path "./datasets/case_resigned_explore_${mode}_sort_by_formula.tsv" \
        --sort_by_value 0 \
        
}

try_formula_mode "diff"
try_formula_mode "ratio"




python ./look_into_data_ver_${ver}.py \
    --function save_frame_of_fourteen_traits \
    --alpha_file_path "./datasets/case_resigned.tsv" \
    --out_file_path "./datasets/case_resigned_var_fourteen.tsv" \


python ./look_into_data_ver_${ver}.py \
    --function save_frame_of_fourteen_traits \
    --alpha_file_path "./datasets/case_master.tsv" \
    --out_file_path "./datasets/case_master_var_fourteen.tsv" \


python ./look_into_data_ver_${ver}.py \
    --function save_priority_frame_for_simple_traits_in_resignation \
    --in_file_path "./datasets/case_master_var_fourteen.tsv" \
    --out_file_path "./datasets/greater_mwu_table_for_resignation.tsv" \


function try_formula_for_fourteen_traits_of_mode () {
    mode=${1}
    python ./look_into_data_ver_${ver}.py \
        --function save_personal_trait_formula_frame_for_fourteen_traits \
        --mode ${mode} \
        --in_file_path "./datasets/case_resigned_var_fourteen.tsv" \
        --out_file_path "./datasets/case_resigned_var_fourteen_explore_${mode}.tsv" \
    
}
try_formula_for_fourteen_traits_of_mode "diff"
try_formula_for_fourteen_traits_of_mode "sum"
try_formula_for_fourteen_traits_of_mode "ratio"


python ./look_into_data_ver_${ver}.py \
    --function save_trait_frame_with_all_binary_trait_columns \
    --in_file_path "./datasets/case_master_var_fourteen.tsv" \
    --out_file_path "./datasets/case_master_var_fourteen_with_binary_traits.tsv" \


python ./look_into_data_ver_${ver}.py \
    --function save_priority_frame_by_mann_whitney_for_binary_traits \
    --in_file_path "./datasets/case_master_var_fourteen_with_binary_traits.tsv" \
    --out_file_path "./datasets/greater_mwu_table_for_resignation_with_binary_traits.tsv" \
    --alternative 'greater' \


python ./look_into_data_ver_${ver}.py \
    --function save_priority_frame_by_mann_whitney_for_binary_traits \
    --in_file_path "./datasets/case_master_var_fourteen_with_binary_traits.tsv" \
    --out_file_path "./datasets/less_mwu_table_for_resignation_with_binary_traits.tsv" \
    --alternative 'less' \


python ./look_into_data_ver_${ver}.py \
    --function save_pvalue_table_of_fisher_exact_in_resigned \
    --in_file_path "./datasets/case_master.tsv" \
    --out_file_path "./datasets/fisher_exact_pvalue_table_for_resignation.tsv" \


python ./look_into_data_ver_${ver}.py \
    --function visualize_fisher_exact_pvalue_table_for_resignation \
    --in_file_path "./datasets/fisher_exact_pvalue_table_for_resignation.tsv" \
    --out_file_path "./datasets/fisher_exact_pvalue_table_for_resignation.pdf" \


function make_contingency_table () {
    theta=${1}
    python ./look_into_data_ver_${ver}.py \
        --function save_contingency_table_of_resigned \
        --in_file_path "./datasets/case_master.tsv" \
        --out_file_path "./datasets/fisher_exact_contingency_table_for_resignation_theta_${theta}.tsv" \
        --theta ${theta} \

}
make_contingency_table 10
make_contingency_table 25



python ./look_into_data_ver_${ver}.py \
    --function save_binary_trait_frame_in_seven_traits \
    --in_file_path "./datasets/case_master.tsv" \
    --out_file_path "./datasets/case_master_with_binary_traits_out_of_seven.tsv" \


function save_upper_careful_frame () {
    local theta=${1}
    python ./look_into_data_ver_${ver}.py \
        --function save_upper_frame_by_threshold_against_carefulness \
        --in_file_path "./datasets/case_master_with_binary_traits_out_of_seven.tsv" \
        --out_file_path "./datasets/upper_carefulness_by_theta_${theta}.tsv" \
        --theta ${theta} \

}
save_upper_careful_frame 25
save_upper_careful_frame 10




function perform_mwu_test_for_carefulness () {
    local theta=${1}
    local side=${2}
    local mode=${3}

    python ./look_into_data_ver_${ver}.py \
        --function save_mann_whitney_priority_frame_of_traits \
        --in_file_path "./datasets/upper_carefulness_by_theta_${theta}.tsv" \
        --out_file_path "./datasets/${side}_mwu_table_for_carefulness_with_theta_${theta}_with_${mode}_traits.tsv" \
        --mode ${mode} \
        --alternative ${side} \

}
# perform_mwu_test_for_carefulness 25 "greater" "seven"
function perform_mwu_test_for_theta () {
    local theta=${1}
    perform_mwu_test_for_carefulness ${theta} "greater" "seven"
    perform_mwu_test_for_carefulness ${theta} "greater" "binary"
    perform_mwu_test_for_carefulness ${theta} "less" "seven"
    perform_mwu_test_for_carefulness ${theta} "less" "binary"
}
perform_mwu_test_for_theta 25
perform_mwu_test_for_theta 10


python ./look_into_data_ver_${ver}.py \
    --function save_reports_on_most_generous_thresholds_on_trait \
    --in_file_path "./datasets/case_master_with_binary_traits_out_of_seven.tsv" \
    --out_file_path "./datasets/nonresigned_traits_with_thresholds.tsv" \



python ./look_into_data_ver_${ver}.py \
    --function play_ground_alpha \

python ./look_into_data_ver_${ver}.py \
    --function play_ground_beta \

python ./look_into_data_ver_${ver}.py \
    --function play_ground_gamma \

python ./look_into_data_ver_${ver}.py \
    --function play_ground_delta \


python ./look_into_data_ver_${ver}.py \
    --function play_ground_zeta \





"""

import itertools, json, tqdm
import pandas as pd
import numpy as np


# Custom imports.
from toolbox import (
        get_diff_frame, 
        select_resigned_frame, 
        select_developing_frame, 
        get_advanced_frame, 
        make_master_table_by_adding_staff_ids, 
        divide_table_into_resigned_and_nonresigned, 
        divide_table_into_developing_and_nondeveloping, 
        select_staff_frame_based_on_trait_threshold, 
        select_staff_frame_based_on_trait_ratio, 
        save_personal_trait_formula_frame_for_seven_traits, 
        get_unit_frame_of_fourteen_traits, 
        get_trait_frame, 
        save_frame_of_fourteen_traits, 

        )
from hypotheses import (
        perform_mann_whitney_test, 
        get_priority_frame_for_traits_by_mann_whitney_test, 
        select_fourteen_traits, 
        save_priority_frame_for_simple_traits_in_resignation, 
        add_binary_trait_values_to_staff_frame, 
        get_binary_trait_label, 
        get_formula_frame_for_trait, 
        get_formula_frame_for_select_traits, 
        save_personal_trait_formula_frame_for_fourteen_traits, 
        get_trait_frame_with_binary_trait_columns, 
        save_trait_frame_with_all_binary_trait_columns, 
        save_binary_trait_frame_in_seven_traits, 
        get_binary_trait_columns, 
        save_priority_frame_by_mann_whitney_for_binary_traits, 
        )
from contingency import (
        get_upper_frame_by_threshold, 
        get_contingency_table_of_resigned_by_threshold, 
        save_pvalue_table_of_fisher_exact_in_resigned, 
        save_contingency_table_of_resigned, 
        save_upper_frame_by_threshold_against_carefulness, 
        select_seven_traits, 
        save_mann_whitney_priority_frame_of_traits, 
        visualize_fisher_exact_pvalue_table_for_resignation, 
        get_pvalue_of_fisher_exact_by_threshold, 
        )




def play_ground_alpha(**kwargs):
    
    in_file_path = "./datasets/case_master_with_binary_traits_out_of_seven.tsv"
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    
    theta = 25
    upper_frame = get_upper_frame_by_threshold(
            staff_frame, 
            column = "慎重性", 
            top_percent = theta, 
            )
    print(upper_frame)
    



def play_ground_beta(**kwargs):
    
    return






def play_ground_gamma(**kwargs):
    """
    Table of probability.
    # 41-45 bin.
    

    """
    from matplotlib import pyplot as plt
    import numpy as np

    in_file_path = "./datasets/case_master_with_binary_traits_out_of_seven.tsv"
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    
    print(staff_frame)
    
    resigned_frame = select_resigned_frame(staff_frame)
    advanced_frame = get_advanced_frame(staff_frame)
    traits_resigned = np.array(resigned_frame["慎重性"])
    traits_advanced = np.array(advanced_frame["慎重性"])
    print(f"{traits_resigned=}")
    print(f"{traits_advanced=}")
    values = range(1, 101, 1)

    counts = resigned_frame["慎重性"].value_counts()
    freqs = counts.tolist()
    measures = counts.index.tolist()

    print(f"{counts=}")
    print(type(counts))
    print(f"{freqs=}")
    print(f"{measures=}")

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture
    from scipy.stats import norm

    #data = np.loadtxt('file.txt') ##loading univariate data.
    data = np.array(resigned_frame["慎重性"])
    gmm = GaussianMixture(n_components = 3).fit(data.reshape(-1, 1))

    plt.figure()
    plt.hist(data, bins=50, histtype='stepfilled', density=True, alpha=0.5)
    plt.xlim(min(data), max(data))
    f_axis = data.copy().ravel()
    f_axis.sort()
    a = []
    for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
        a.append(weight*norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
        plt.plot(f_axis, a[-1])
    plt.plot(f_axis, np.array(a).sum(axis = 0), 'k-')
    plt.xlabel('Variable')
    plt.ylabel('PDF')
    plt.tight_layout()
    plt.show()


    """
    plt.hist(traits_resigned, bins = 100)
    plt.show()

    # "gmr" Python library?
    from sklearn.mixture import GaussianMixture

    traits = np.array(staff_frame["慎重性"])
    data = traits.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0).fit(data)
    """

    return 


def save_reports_on_most_generous_thresholds_on_trait(
        in_file_path = "./datasets/case_master_with_binary_traits_out_of_seven.tsv", 
        out_file_path = "./datasets/nonresigned_traits_with_thresholds.tsv", 
        debug = False, **kwargs
        ):
    """
    Mine the thresholds with < 5% probability of early resignation.
    Thresholding is based on value instead of percentage.
    """
    advances_greater = [
            "[活動性]-minus-[社交性]", 
            "[固執性]-over-[社交性]", 
            "[新奇性]-minus-[社交性]", 
            "[活動性]-over-[社交性]", 
            "[固執性]-minus-[社交性]", 
            "[決断性]-over-[社交性]", 
            "[新奇性]-over-[社交性]", 
            "[決断性]-minus-[社交性]", 
            "[主体性]-over-[社交性]", 
            "[固執性]-over-[活動性]", 
            "[慎重性]-over-[社交性]", 
            "[主体性]-minus-[社交性]", 
            "[固執性]-minus-[活動性]", 
            ]
    advances_less = [
            "[社交性]-plus-[慎重性]", 
            "[社交性]-minus-[活動性]", 
            "[社交性]-over-[固執性]", 
            "[社交性]-minus-[新奇性]", 
            "[社交性]-over-[活動性]", 
            "[社交性]-minus-[固執性]", 
            "[活動性]-plus-[社交性]", 
            "[社交性]-plus-[活動性]", 
            "[社交性]-over-[決断性]", 
            "[社交性]-over-[新奇性]", 
            "[社交性]-minus-[決断性]", 
            "[社交性]-plus-[固執性]", 
            "[固執性]-plus-[社交性]", 
            "[社交性]-plus-[新奇性]", 
            "[新奇性]-plus-[社交性]", 
            "[社交性]-over-[主体性]", 
            "[慎重性]-plus-[活動性]", 
            "[活動性]-plus-[慎重性]", 
            "[社交性]-plus-[主体性]", 
            "[主体性]-plus-[社交性]", 
            "[活動性]-over-[固執性]", 
            "[社交性]-over-[慎重性]", 
            "[社交性]-minus-[主体性]", 
            "[活動性]-minus-[固執性]", 
            "[決断性]-plus-[社交性]", 
            "[社交性]-plus-[決断性]", 
            ]
    staff_frame = pd.read_csv(in_file_path, sep = "\t")

    def get_rate_of_resigned_per_trait_per_theta(
            trait, theta, staff_frame, 
            side = "greater", 
            debug = False, ):
        resigned_frame = select_resigned_frame(staff_frame)
        if side == "greater": 
            trait_staff_frame = staff_frame[staff_frame[trait] > theta]
            trait_resigned_frame = resigned_frame[resigned_frame[trait] > theta]
        elif side == "less": 
            trait_staff_frame = staff_frame[staff_frame[trait] < theta]
            trait_resigned_frame = resigned_frame[resigned_frame[trait] < theta]
        else:
            return None
        
        rate_of_resignation = 0.0
        count_of_total = len(trait_staff_frame)
        count_of_resigned = len(trait_resigned_frame)
        if len(trait_staff_frame) != 0:
            rate_of_resignation = float(count_of_resigned) / count_of_total
        
        report = dict(trait = trait, side = side)
        report["threshold"] = theta
        report["resigned_rate"] = rate_of_resignation
        report["count_of_total"] = count_of_total
        report["count_of_resigned"] = count_of_resigned

        if debug:
            print(len(trait_resigned_frame), len(trait_staff_frame))
            print(rate_of_resignation)
        return report
    
    def find_most_generous_threshold_on_trait_per_side(
            trait, 
            staff_frame, 
            side = "greater", 
            ):
        thetas = list(set(staff_frame[trait].tolist()))
        thetas = sorted(thetas, reverse = (side == "less"))
        #print(thetas)
        # Select the report with the optimal threshold.
        for i, theta in enumerate(thetas):
            report = get_rate_of_resigned_per_trait_per_theta(
                    trait, theta, 
                    staff_frame, 
                    side = side, 
                    )
            if report["resigned_rate"] < 0.05: 
                report["count_of_search"] = i + 1
                break
        return report
    
    def get_reports_on_most_generous_thresholds_on_trait(
            staff_frame, 
            traits, 
            side = "greater", 
            ):
        reports = list()
        for trait in traits:
            report = find_most_generous_threshold_on_trait_per_side(
                    trait, 
                    staff_frame, 
                    side = side, 
                    )
            #print(report)
            reports.append(report)
        return reports
    
    reports_greater = get_reports_on_most_generous_thresholds_on_trait(
            staff_frame, 
            traits = advances_greater, 
            side = "greater",
            )
    reports_less = get_reports_on_most_generous_thresholds_on_trait(
            staff_frame, 
            traits = advances_less, 
            side = "less",
            )
    reports = reports_greater + reports_less
    frame = pd.DataFrame(reports)
    if debug:
        print(json.dumps(reports_greater, indent = 4, ensure_ascii = False))
        print(frame)
    frame.to_csv(out_file_path, sep = "\t", index = False)
    return frame


def play_ground_zeta(**kwargs):
    in_file_path = "./datasets/upper_carefulness_by_theta_25.tsv"
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    print(staff_frame)


    return


def play_ground_delta(**kwargs):
    """
    # Perform this on smaller sample sizes.
    """
    in_file_path = "./datasets/case_master_with_binary_traits_out_of_seven.tsv"
    staff_frame = pd.read_csv(in_file_path, sep = "\t")
    less_thetas = {
            "[社交性]-plus-[慎重性]": 86, 
            "[社交性]-minus-[活動性]": -8, 
            "[社交性]-over-[固執性]": 0.861538461538462, 
            "[社交性]-minus-[新奇性]": -19, 

            "[社交性]-over-[活動性]": 0.84, 
            "[社交性]-minus-[固執性]": -9, 
            "[活動性]-plus-[社交性]": 80, 
            "[社交性]-over-[決断性]": 0.706896551724138, 
            "[社交性]-over-[新奇性]": 0.685185185185185, 
            "[社交性]-minus-[決断性]": -18, 
            "[社交性]-plus-[固執性]": 71, 
            "[社交性]-plus-[新奇性]": 80, 
            "[社交性]-over-[主体性]": 0.75, 
            "[慎重性]-plus-[活動性]": 100, 
            "[社交性]-plus-[主体性]": 79, 
            "[活動性]-over-[固執性]": 0.701492537313433, 
            "[社交性]-over-[慎重性]": 0.585714285714286, 
            "[社交性]-minus-[主体性]": -14, 
            "[活動性]-minus-[固執性]": -20, 
            "[決断性]-plus-[社交性]": 68, 
            }
    """
    [社交性]-plus-[慎重性]	less	86
    [社交性]-minus-[活動性]	less	-8
    [社交性]-over-[固執性]	less	0.861538461538462
    [社交性]-minus-[新奇性]	less	-19

    [社交性]-over-[活動性]	less	0.84
    [社交性]-minus-[固執性]	less	-9
    [活動性]-plus-[社交性]	less	80
    [社交性]-plus-[活動性]	less	80
    [社交性]-over-[決断性]	less	0.706896551724138
    [社交性]-over-[新奇性]	less	0.685185185185185
    [社交性]-minus-[決断性]	less	-18
    [社交性]-plus-[固執性]	less	71
    [固執性]-plus-[社交性]	less	71
    [社交性]-plus-[新奇性]	less	80
    [新奇性]-plus-[社交性]	less	80
    [社交性]-over-[主体性]	less	0.75
    [慎重性]-plus-[活動性]	less	100
    [活動性]-plus-[慎重性]	less	100
    [社交性]-plus-[主体性]	less	79
    [主体性]-plus-[社交性]	less	79
    [活動性]-over-[固執性]	less	0.701492537313433
    [社交性]-over-[慎重性]	less	0.585714285714286
    [社交性]-minus-[主体性]	less	-14
    [活動性]-minus-[固執性]	less	-20
    [決断性]-plus-[社交性]	less	68
    [社交性]-plus-[決断性]	less	68

    """
    traits = list(less_thetas.keys())
    print(traits)
    
    trait = traits[0]

    in_file_path = "./datasets/upper_carefulness_by_theta_25.tsv"
    target_frame = pd.read_csv(in_file_path, sep = "\t")

    #advanced_frame = get_advanced_frame(staff_frame)

    staff_groups = dict()
    for trait in traits:
        count_frame = target_frame[target_frame[trait] < less_thetas[trait]]
        staffs = count_frame["staff"].tolist()
        staff_groups[trait] = staffs
    
    print(staff_groups)
    staffs_total = target_frame["staff"].tolist()
    marks = dict()
    for staff in staffs_total:
        marks[staff] = [trait for trait in traits if staff in staff_groups[trait]]
    
    print(marks)
    counts = {key: len(marks[key]) for key in list(marks.keys())}
    print(counts)
    from collections import Counter

    print(Counter(list(counts.values())))
    print(len(target_frame))

    return










if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='test code.')
    parser.add_argument('--function', '-f', nargs='?', type=str, default=None, help='function to be called')
    parser.add_argument('--in_file_path', '-i', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--mid_file_path', '-m', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--out_file_path', '-o', nargs='?', type=str, default=None, help='help.', required=False)

    parser.add_argument('--zip_file_path', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--query_file_path', nargs='?', type=str, default=None, help='help.', required=False)
    
    parser.add_argument('--alpha_file_path', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--beta_file_path', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--sort_by_value', nargs='?', type=int, default=0, help='help.', required=False)
    parser.add_argument('--mode', nargs='?', type=str, default="diff", help='help.', required=False)
    parser.add_argument('--alternative', nargs='?', type=str, default="greater", help='help.', required=False)
    parser.add_argument('--theta', nargs='?', type=int, default=10, help='help.', required=False)
    

    args = parser.parse_args()

    eval(args.function)(
            **args.__dict__
            )

# End of document
