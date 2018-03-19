#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import os
os.chdir("/Users/rebecagonzalez/Desktop/DeepLearning/ud120-projects/datasets_questions")

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


print('Number of people in the Enron dataset: {0}'.format(len(enron_data)))

print('Number of columnos per person in the Enron dataset: {0}'.format(len(enron_data.values()[0])))


'''In other words, count the number of entries in the dictionary where
data[person_name]["poi"]==1'''

pois = [x for x, y in enron_data.items() if y['poi']]
print 'Number of POI\'s: {0}'.format(len(pois))

# DELETE ME
print(enron_data['PRENTICE JAMES'])


salary_count = 0
for key, value in enron_data.items():
    if value['director_fees'] != "NaN":
        salary_count += 1;
        print(value['email_address'],value['director_fees'])
print("The number of people whose director fees is not null: %f " %salary_count)



email_count = 0
for key, value in enron_data.items():
    if value['email_address'] != "NaN":
        email_count += 1;
print("The number of people whose email is not null: %f" %email_count)

print(enron_data['COLWELL WESLEY'])




print()
print()

names = ['SKILLING JEFFREY K', 'FASTOW ANDREW S', 'LAY KENNETH L']
names_payments = {name:enron_data[name]['total_payments'] for name in names}
print sorted(names_payments.items(), key=lambda x: x[1], reverse=True)