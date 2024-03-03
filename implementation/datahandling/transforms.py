from torch import FloatTensor, IntTensor

def to_numeric_label(in_label, target_labels):
    in_label = in_label.split("|")
    return FloatTensor([1 if (x in in_label) else 0 for x in target_labels])

def to_class_int(in_label, target_labels):
    in_label = in_label.split("|")
    numerics = [1 if (x in in_label) else 0 for x in target_labels]
    value=0
    if(sum(numerics)) > 0:
        value = numerics.index(1)
    return IntTensor([value])
    