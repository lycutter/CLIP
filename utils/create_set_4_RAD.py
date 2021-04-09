import json

pathfile = 'E:\Datasets\VQA-Rad/VQA_RAD Dataset Public.json'
data_records = json.load(open(pathfile, "r"))
test_set = []
train_set = []
count = 0

for record in data_records:
    sample = {}
    count += 1
    sample['qid'] = count
    sample['image_name'] = record['image_name']
    sample['image_organ'] = record['image_organ']
    sample['answer'] = record['answer']
    sample['answer_type'] = record['answer_type']
    sample['question_type'] = record['question_type']
    if "freeform" in record['phrase_type']:
        sample['question'] = record['question']
        sample['phrase_type'] = "freeform"
    elif "para" in record['phrase_type']:
        sample['question'] = record['question']
        sample['phrase_type'] = "para"
    if "test" in record['phrase_type']:
        test_set.append(sample.copy())
    else:
        train_set.append(sample.copy())
    if record['question_frame'] != 'NULL':
        count += 1
        sample['qid'] = count
        sample['question'] = record['question_frame']
        sample['phrase_type'] = "frame"
        train_set.append(sample.copy())
with open('G:\PythonWorkplace\VQA-master\data/trainset_rad.json', 'w') as outfile:
    json.dump(train_set, outfile)
with open('G:\PythonWorkplace\VQA-master\data/valset_rad.json', 'w') as outfile:
    json.dump(test_set, outfile)