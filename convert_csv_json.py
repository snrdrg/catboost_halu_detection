import csv
import json

csv_name = 'eng-cissp-4k.csv'
json_name = 'eng-cissp-4k.json'

with open(csv_name, mode='r', encoding='utf-8-sig') as csv_file:

    reader = csv.DictReader(csv_file)
    with open(json_name, 'w', encoding='utf-8-sig') as json_file:
        json.dump({'data': [x for x in reader]}, json_file)

with open(json_name, 'r', encoding='utf-8-sig') as json_file:
    json_data = json.load(json_file)

print('complete')
