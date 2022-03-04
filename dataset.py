#One file for the input and one file for the target
#seperated by newline 
import math
import os
import csv
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tdfs
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


#Split of test, train and validation:
total_samples = 38269
n_train = math.floor(total_samples * 0.95)
n_test = math.floor((total_samples-n_train)*0.6)
n_val = total_samples - n_train - n_test


def split(filehandler, delimiter=','):
    reader = csv.reader(filehandler, delimiter=delimiter)

    train_outputpath = 'train.csv'
    test_outputpath = 'test.csv'
    val_outputpath = 'val.csv'

    train_writer = csv.writer(open(train_outputpath, 'w'), delimiter=delimiter)
    test_writer = csv.writer(open(test_outputpath, 'w'), delimiter=delimiter)
    val_writer = csv.writer(open(val_outputpath, 'w'), delimiter=delimiter)


    headers = next(reader)
    train_writer.writerow(headers)
    test_writer.writerow(headers)
    val_writer.writerow(headers)

    for i, row in enumerate(reader):
        if i < n_train:
            train_writer.writerow(row)

        elif n_train <= i < n_train + n_test:
            test_writer.writerow(row)
        
        elif n_train + n_test <= i < total_samples:
            val_writer.writerow(row)


def get_data(filename):
    train_reader = csv.DictReader(open(filename, 'r'))

    print(filename)
    questions = []
    answers = []

    for  line in train_reader:
        questions = np.append(questions, line['Question'])
        answers = np.append(answers, line['Answer'])
    

    #write questions and answers to separate csv-files
    np.savetxt(filename[:-4] + 'questions.csv', questions, fmt='%s')
    np.savetxt(filename[:-4] + 'answers.csv', answers, fmt='%s')

    return questions, answers    


def get_data_tf():
    ds_test = tf.data.TextLineDataset('testquestions.csv')

    #split into questions and answers
    for line in ds_test.take(5):
        print(line.numpy().decode('utf-8'))


if __name__ == "__main__":

    get_data('test.csv')
    get_data('val.csv')
    get_data('train.csv')

 