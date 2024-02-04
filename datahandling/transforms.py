def to_numeric_label(in_label):
    in_label = in_label.split("|")
    return FloatTensor([1 if (x in in_label) else 0 for x in labels])