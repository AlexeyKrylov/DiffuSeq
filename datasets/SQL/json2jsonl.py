import json


def json2jsonl(filename1, writename):
    with open(filename1, "r", encoding='utf8') as f:
        file1 = json.load(f)

    with open(writename, 'w', encoding='utf8') as f:
        for i in file1:
            d = dict()
            d['src'] = " ".join(i["question_toks"]).strip()
            d['trg'] = " ".join(i["query_toks"]).strip()
            json.dump(d, f)
            f.write('\n')

if __name__ == '__main__':
    json2jsonl("datasets/SQL/en_iid_train.json",
              "datasets/SQL/train.jsonl")
    json2jsonl("datasets/SQL/en_iid_test.json",
              "datasets/SQL/valid.jsonl")