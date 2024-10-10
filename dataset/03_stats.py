import json


if __name__ == "__main__":
    stats = {
        'no': 0,
        'address': 0,
        'company': 0,
        'total': 0,
        'date': 0
    }

    total = 0
    subtotal = 0

    for s in ['train', 'test', 'valid']:
        with open('ents_' + s + '.jsonl', 'r') as f:
            data = f.readlines()

        for d in data:
            d = json.loads(d)
            total += 1
            for ent in d['labels']:
                subtotal += 1
                stats[ent[2]] += 1

    for k, v in stats.items():
        print(k, "->", "{:.3f}".format(v/total))

    print("OVR: ", "{:.3f}".format(subtotal/(5*total)))
   

