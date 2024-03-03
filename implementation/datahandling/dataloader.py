import pandas as pd

def get_list_from_txt(path):
    '''
    Reads a .txt file and returns an array containing every line. Newline characters are stripped
    '''
    with open(path) as f:
        return [x.rstrip() for x in f.readlines()]

def extract_annotation_targets(annotations, filter_key, targets):
    '''
    Extracts target values from an annotations DataFrame using targets for values of the column related to the filter_key
    '''
    return annotations[annotations[filter_key].isin(targets)]

def extract_unique_labels(annotations, label_key):
    labels = set([x for y in [x.split("|") for x in annotations[label_key]] for x in y])
    labels.remove("No Finding")
    return labels

def extract_n_single_label_images(annotations_file, n, target_file):
    annotations = pd.read_csv(annotations_file)
    annotations = annotations[annotations["Finding Labels"].str.contains("\|") == False].tail(n)
    annotations.to_csv(target_file, index=False)
    