import json


def txt2jsonl(filename1, filename2, writename):
    file1 = open(filename1, 'r', encoding='utf8').readlines()
    file2 = open(filename2, 'r', encoding='utf8').readlines()

    with open(writename, 'w', encoding='utf8') as f:
        for i, j in zip(file1, file2):
            d = dict()
            d['src'] = i.strip()
            d['trg'] = j.strip()
            json.dump(d, f)
            f.write('\n')

if __name__ == '__main__':
    txt2jsonl("datasets/SparQL/src-english_test_split.txt", "datasets/SparQL/spec-english_test_split.txt", "datasets/SparQL/test.jsonl")
    txt2jsonl("datasets/SparQL/src-english_train_split.txt", "datasets/SparQL/spec-english_train_split.txt",
              "datasets/SparQL/train.jsonl")
    txt2jsonl("datasets/SparQL/src-english_dev_split.txt", "datasets/SparQL/spec-english_dev_split.txt",
              "datasets/SparQL/valid.jsonl")