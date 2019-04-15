from typing import List



def clef2019_average_precision(gold_labels:List[int], ranked_lines:List[int]):

    assert len(gold_labels)==len(ranked_lines)
    assert max(gold_labels)==1 and min(gold_labels)==0 # must be binary labels

    precisions = []
    TP_rank = 0
    num_positive = sum(gold_labels)

    for rank, line_number in enumerate(ranked_lines):
        is_true_positive = gold_labels[line_number] == 1
        if is_true_positive:
            TP_rank += 1
            precision = TP_rank / (rank + 1)
            precisions.append(precision)

    if len(precisions)>0:
        avg_prec = sum(precisions) / num_positive
    else:
        avg_prec = 0.0

    return avg_prec


