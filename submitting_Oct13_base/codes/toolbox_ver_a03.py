
import pandas as pd


# Primitive functions.
def select_resigned_frame(tab_frame):
    frame = tab_frame[tab_frame["1年以内退職"] == 1]
    frame = frame.reset_index(drop = True)
    return frame


def select_developing_frame(tab_frame):
    frame = tab_frame[tab_frame["活躍できない"] == 1]
    frame = frame.reset_index(drop = True)
    return frame


def get_diff_frame(tab_frame, resigned_frame, column = "No."):
    resigned_ids = [str(atom) for atom in resigned_frame[column].tolist()]
    nonresigned_frame = tab_frame[~tab_frame.apply(lambda x: str(x[column]) in resigned_ids, axis=1)]
    frame = nonresigned_frame.reset_index(drop = True)
    return frame


def get_advanced_frame(staff_frame, debug = False):
    # Advanced := NOT resigned and NOT developing.
    resigned_frame = select_resigned_frame(staff_frame)
    nonresigned_frame = get_diff_frame(staff_frame, resigned_frame, column = "staff")
    developing_frame = select_developing_frame(staff_frame)
    advanced_frame = get_diff_frame(nonresigned_frame, developing_frame, column = "No.")
    if debug:
        print(f"{len(staff_frame)=}")
        print(f"{len(resigned_frame)=}")
        print(f"{len(developing_frame)=}")
        print(f"{len(advanced_frame)=}")
        overlap_frame = select_developing_frame(resigned_frame)
        print(f"{len(overlap_frame)=}")
    return advanced_frame


def get_trait_frame(staff_frame):
    submit_frame = staff_frame.iloc[:, 0:13].copy()
    return submit_frame


def select_seven_traits(staff_frame):
    traits = list(staff_frame.columns[6:13])
    return traits


def get_binary_trait_columns(columns):
    atoms = list()
    for column in columns:
        str_list = column.split("-")
        if len(str_list) > 1:
            if str_list[1] in ["over", "minus", "plus"]:
                atoms.append(column)
    return atoms

"""
def get_seven_frame(staff_frame):
    submit_frame = staff_frame.iloc[:, 6:13].copy()
    return submit_frame
"""


class Staff_Table:
    def __init__(
            self, 
            in_file_path = "./datasets/case_master.tsv"):
        class Path: pass
        self.path = Path()
        self.path.to_master_file = in_file_path

        self.frame = pd.read_csv(in_file_path, sep = "\t")
        
    def get_resigned_frame(self):
        return select_resigned_frame(self.frame)

    def get_developing_frame(self):
        return select_developing_frame(self.frame)

    def get_diff(self, frame):
        return get_diff_frame(self.frame, frame, column = "No.")

    def get_developing_frame(self):
        return get_advanced_frame(self.frame)

    def get_trait_frame(self):
        return get_trait_frame(self.frame)

    def get_seven_traits(self):
        return select_seven_traits(self.frame)
    
    def get_binary_traits(self):
        return get_binary_trait_columns(self.frame)




def make_master_table_by_adding_staff_ids(
        in_file_path = "./datasets/table_for_analysis.csv", 
        out_file_path = "./datasets/case_master.tsv", 
        debug = False, **kwargs):
    tab_frame = pd.read_csv(in_file_path, sep = ",")
    header = tab_frame.columns.tolist()
    tab_frame["staff"] = ["Staff_%03d" % (int(num)) for num in tab_frame["No."].tolist()]
    columns = ["staff"] + header
    
    if debug: 
        print(tab_frame)
        print(columns)
    
    tab_frame = tab_frame[columns]
    tab_frame.to_csv(out_file_path, index = False, sep = "\t")
    return tab_frame


def divide_table_into_resigned_and_nonresigned(
        in_file_path = "./datasets/table_for_analysis.csv", 
        alpha_file_path = "./datasets/case_resigned.tsv", 
        beta_file_path = "./datasets/case_nonresigned.tsv", 
        debug = False, **kwargs):

    tab_frame = pd.read_csv(in_file_path, sep = "\t")
    resigned_frame = select_resigned_frame(tab_frame)
    nonresigned_frame = get_diff_frame(tab_frame, resigned_frame, column = "No.")
    
    if debug:
        print(tab_frame)
        print(f"{len(tab_frame)=}")
        print(tab_frame.columns)
        print(resigned_frame)
        print(f"{len(resigned_frame)=}")
        print(f"{len(nonresigned_frame)=}")
    resigned_frame.to_csv(alpha_file_path, index = False, sep = "\t")
    nonresigned_frame.to_csv(beta_file_path, index = False, sep = "\t")
    return resigned_frame


def divide_table_into_developing_and_nondeveloping(
        in_file_path = "./datasets/table_for_analysis.csv", 
        alpha_file_path = "./datasets/case_developing.tsv", 
        beta_file_path = "./datasets/case_nondeveloping.tsv", 
        debug = False, **kwargs):

    tab_frame = pd.read_csv(in_file_path, sep = "\t")
    developing_frame = select_developing_frame(tab_frame)
    nondeveloping_frame = get_diff_frame(tab_frame, developing_frame, column = "No.")
    
    developing_frame.to_csv(alpha_file_path, index = False, sep = "\t")
    nondeveloping_frame.to_csv(beta_file_path, index = False, sep = "\t")
    return developing_frame


def select_staff_frame_based_on_trait_threshold(
        trait_a = "社交性", 
        trait_b = "慎重性", 
        staff_frame = None, debug = False):
    frame = staff_frame[
            (staff_frame[trait_a] < 50)
            & (50 < staff_frame[trait_b])
            ].copy()
    frame["diff"] = staff_frame[trait_b] - staff_frame[trait_a]
    frame = frame.sort_values("diff", ascending = False)
    frame = frame.reset_index(drop = True)
    if debug: print(frame[["staff", trait_a, trait_b]])
    return frame


def select_staff_frame_based_on_trait_ratio(
        trait_a = "社交性", 
        trait_b = "慎重性", 
        staff_frame = None, debug = False):
    staff_frame["ratio"] = staff_frame[trait_b] / staff_frame[trait_a]
    frame = staff_frame[
            (staff_frame["ratio"] > 1.4)
            ].copy()
    frame = frame.sort_values("ratio", ascending = False)
    frame = frame.reset_index(drop = True)
    if debug: 
        print(staff_frame)
        print(frame[["staff", trait_a, trait_b]])
    return frame


def save_personal_trait_formula_frame_for_seven_traits(
        mode = "diff", 
        alpha_file_path = "./datasets/case_resigned.tsv", 
        out_file_path = "./datasets/case_resigned_explore_diff.tsv", 
        sort_by_value = 0, 
        seven_traits = [
                "活動性", "社交性", "慎重性", "新奇性", 
                "固執性", "主体性", "決断性", 
                ], 
        debug = False, **kwargs):
    """
    # Define HIGH/LOW by a threshold 50.
    # Define differences [活動性]-minus-[社交性] and ratios [活動性]-over-[社交性].
    """
    def get_personal_trait_score_frame_for_formula(
            trait_a = "社交性", 
            trait_b = "慎重性", 
            staff_frame = None, 
            mode = "diff", 
            ):
        if mode == "diff":
            label = "[%s]-minus-[%s]" % (trait_b, trait_a)
            score_frame = select_staff_frame_based_on_trait_threshold(
                    trait_a = trait_a, 
                    trait_b = trait_b, 
                    staff_frame = staff_frame, 
                    )
        elif mode == "ratio":
            label = "[%s]-over-[%s]" % (trait_b, trait_a)
            score_frame = select_staff_frame_based_on_trait_ratio(
                    trait_a = trait_a, 
                    trait_b = trait_b, 
                    staff_frame = staff_frame, 
                    )
        else:
            label = "[%s]-XXX-[%s]" % (trait_b, trait_a)
            score_frame = pd.DataFrame()

        header = [mode, "staff", trait_b, trait_a]
        temp_frame = score_frame[header].copy()
        temp_frame["label"] = [label] * len(temp_frame)
        formula_frame = temp_frame[["label"] + header]

        return label, formula_frame
    
    def get_formula_frame_for_seven_traits(
            mode = "diff", 
            traits = [
                    "活動性", "社交性", "慎重性", "新奇性", 
                    "固執性", "主体性", "決断性", 
                    ], 
            staff_frame = None, 
            debug = False, ):
        
        
        lex_of_staffs = dict()
        for trait_a in traits:
            for trait_b in traits:
                if trait_a != trait_b:
                    label, formula_frame = get_personal_trait_score_frame_for_formula(
                            trait_a = trait_a, 
                            trait_b = trait_b, 
                            staff_frame = staff_frame, 
                            mode = mode, 
                            )

                    lex_of_staffs[label] = formula_frame
                    if debug:
                        print(score_frame[["staff", trait_a, trait_b]])
        submit_frame = pd.concat(list(lex_of_staffs.values()), ignore_index = True)
        return submit_frame
    
    staff_frame = pd.read_csv(alpha_file_path, sep = "\t")
    submit_frame = get_formula_frame_for_seven_traits(
            mode = mode, 
            traits = seven_traits, 
            staff_frame = staff_frame, 
            )

    if sort_by_value > 0:
        submit_frame = submit_frame.sort_values(mode, ascending = False)
        submit_frame = submit_frame.reset_index(drop = True)
    submit_frame.to_csv(out_file_path, sep = "\t", index = False)
    return submit_frame


def get_unit_frame_of_fourteen_traits(trait = "活動性", staff_frame = None):
    trait_values = staff_frame[trait]
    weaks = 50 - trait_values
    strongs = trait_values - 50
    # Set negative values to zero.
    def set_zero(atoms): 
        atoms[atoms < 0] = 0
        return atoms
    weaks = set_zero(weaks); strongs = set_zero(strongs); 
    frame = pd.DataFrame({
            ("未・" + trait): weaks.tolist(), 
            ("既・" + trait): strongs.tolist(), 
            })
    return frame


def save_frame_of_fourteen_traits(
        alpha_file_path = "./datasets/case_resigned.tsv", 
        out_file_path = "./datasets/case_resigned_var_fourteen.tsv", 
        seven_traits = [
                "活動性", "社交性", "慎重性", "新奇性", 
                "固執性", "主体性", "決断性", 
                ], 
        debug = False, **kwargs):
    staff_frame = pd.read_csv(alpha_file_path, sep = "\t")
    # Columns 1-6: control fields, 7-13: the seven traits.

    submit_frame = get_trait_frame(staff_frame)
    for trait in seven_traits:
        temp_frame = get_unit_frame_of_fourteen_traits(trait, staff_frame)
        submit_frame = pd.concat([submit_frame, temp_frame], axis = 1)
    if debug: print(submit_frame)
    submit_frame.to_csv(out_file_path, sep = "\t", index = False)
    return submit_frame














