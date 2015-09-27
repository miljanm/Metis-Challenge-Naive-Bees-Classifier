import csv
import datetime


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