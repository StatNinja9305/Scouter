

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


def reporting_failed_attempt(**kwargs):
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

