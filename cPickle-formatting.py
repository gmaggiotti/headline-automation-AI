from six.moves import cPickle as pickle
import numpy as np
import os
import io

folder ="data-es/tn/"

def maybe_extract(folder, force=False):
    root = os.path.splitext(os.path.splitext(folder)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, folder))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
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

datasets = maybe_pickle(dataset_folders)


#######################

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class+tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


pickle_file = os.path.join(data_root, 'notMNIST.pickle')
