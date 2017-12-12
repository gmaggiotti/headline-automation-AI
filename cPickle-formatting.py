from six.moves import cPickle as pickle
import numpy as np
import os
import io

folder ="data-es/tn/sports-50k"

def maybe_extract(folder, force=False):
    root = os.path.splitext(os.path.splitext(folder)[0])[0]  # remove .tar.gz
    data_folders = [
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))]
    print(data_folders)
    return data_folders

dataset_folders = maybe_extract(folder)
##############

def load_letter(folder):
    """Load the data for a single letter label."""
    txt_files = os.listdir(folder)
    headers = ['']*len(txt_files)
    descs = ['']*len(txt_files)
    keywords = ['']*len(txt_files)

    print(folder)
    num_files = 0
    for txt_file in txt_files:
        try:
            file = open(folder+'/'+txt_file, 'r')
            line_num = 0
            header = ''
            desc = ''
            for line in file:
                if line_num == 0:
                    header = line
                else:
                    desc += line
                line_num+=1

            headers[num_files] = header
            descs[num_files] = desc
            keywords[num_files] = ''
            num_files+=1

        except IOError as e:
            print('Could not read, skipping.')
    dataset = [headers,descs,keywords]
    print('Full dataset for:', folder)
    return dataset


####################

"""Save data from a set of single letter into a .piclke file. Iterates for each letter"""
def maybe_pickle(data_folders, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + '.pkl'
        dataset_names.append(set_filename)

        dataset = load_letter(folder)
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

    return dataset_names

datasets = maybe_pickle([folder])

