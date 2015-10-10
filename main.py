import os
import pdb
import csv
import pandas
import datetime
import numpy as np
from skimage import io
from itertools import islice
# try:
#     import cPickle as pickle
# except:
#     import pickle
import bloscpack as bp

__author__ = 'miljan'


def write_output(predictions):
    """
    Write into csv format required for submission
    :param predictions: 2D list, each row is id, probability
    """
    with open('./output/' + str(datetime.datetime.now())) as output:
        csv_writer = csv.writer(output)
        # write title
        csv_writer.writerow('id,genus')

        for row in predictions:
            csv_writer.writerow(row)


def read_data(type, is_pickled=False):
    """
    Read images from jpg and labels from csv
    :param type: 'train' or 'test'
    :param is_pickled: boolean
    :return:
    """
    pairs = {}
    images = os.listdir('./data/images/' + type)
    data = np.zeros((len(images), 200, 200, 3))
    ids = np.zeros((len(images)))
    labels = np.zeros((len(images)))

    print 'Reading labels...'

    if type == 'train':
        with open('./data/train_labels.csv') as file:
            csv_reader = csv.reader(file)
            csv_reader.next()
            for line in csv_reader:
                pairs[int(line[0])] = line[1]
    print '\tDone'
    print 'Reading images...'

    for i, filename in enumerate(images):
        id = int(filename.split('.')[0])
        img = io.MultiImage('./data/images/' + type + '/' + filename)[0]

        # ignore last dimension
        if img.shape[2] > 3:
            img = img[:, :, :3]

        data[i, :, :, :] = img
        ids[i] = id

        if type == 'train':
            labels[i] = pairs[id]

    print '\tDone'
    print 'Pickling...'

    if is_pickled:
        print '\tDumping data'
        bp.pack_ndarray_file(data, './data/pickles/' + type + '_data.pickle')
        # pickle.dump(data, open('./data/pickles/' + type + '_data.pickle', 'wb'))
        print '\tDumping ids'
        bp.pack_ndarray_file(ids, './data/pickles/' + type + '_ids.pickle')
        # pickle.dump(ids, open('./data/pickles/' + type + '_ids.pickle', 'wb'))
        if type == 'train':
            print '\tDumping labels'
            bp.pack_ndarray_file(labels, './data/pickles/' + type + '_labels.pickle')
            # pickle.dump(labels, open('./data/pickles/' + type + '_labels.pickle', 'wb'))

    print '\tDone'
    return data


if __name__ == '__main__':
    read_data('test', is_pickled=True)