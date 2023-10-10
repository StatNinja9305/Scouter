"""


# Env:
conda create -n P9A --clone Python3.9.1
conda activate P9A

# Installation:
pip install -U pip
pip install pandas numpy matplotlib scipy



# Command:

conda activate P9A


ver=a29


"""


from toolbox import (
        make_master_table_by_adding_staff_ids,
        )

"""
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


from rectangle import (
        save_rectangles_for_50_percent_risk_zone, 
        )
"""
python ./look_into_data_ver_${ver}.py \
    --function save_rectangles_for_50_percent_risk_zone \
    --in_file_path "./datasets/case_master.tsv" \
    --out_file_path "./datasets/rectangles.tsv" \

"""


"""
# Installing Japanese fonts.

sudo apt update
sudo apt install fontconfig

# Check the installed version.
fc-list -V
fc-list | grep 'IPA'

# Install fonts.
sudo apt install fonts-ipaexfont

# Check it.
fc-list | grep 'IPA'

# Edit matplotlibrc.
site-packages/matplotlib/mpl-data/matplotlibrc
# Replace "#font.family : sans-serif" with "font.family : IPAexGothic"
# '#' has to be deleted.

# This case: /home/user/anaconda3/envs/P9A/lib/python3.9/site-packages/

# If not working, delete cache file (other file names are possible):
# $ ls ~/.cache/matplotlib/
# fontlist-v330.json



"""


from rectangle import (
        save_plot_rectangles, 
        )
"""
python ./look_into_data_ver_${ver}.py \
    --function save_plot_rectangles \
    --in_file_path "./datasets/rectangles.tsv" \
    --out_dir_path "./datasets/rectangles/" \

"""



def play_ground_alpha(**kwargs):
    return 



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


from failed_flowchart import save_reports_on_most_generous_thresholds_on_trait




def play_ground_zeta(**kwargs):
    
    return








if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='test code.')
    parser.add_argument('--function', '-f', nargs='?', type=str, default=None, help='function to be called')
    parser.add_argument('--in_file_path', '-i', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--mid_file_path', '-m', nargs='?', type=str, default=None, help='help.', required=False)
    parser.add_argument('--out_file_path', '-o', nargs='?', type=str, default=None, help='help.', required=False)
    
    parser.add_argument('--out_dir_path', nargs='?', type=str, default=None, help='help.', required=False)

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
