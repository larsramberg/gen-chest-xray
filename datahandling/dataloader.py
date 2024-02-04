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