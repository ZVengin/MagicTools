import json


def read_json_file(path):
    with open(path,'r') as f:
        return json.load(f)


def read_jsonl_file(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_text_file(path):
    with open(path,'r') as f:
        data = f.readlines()
    return data


def write_text_file(data, path):
    with open(path, 'w') as f:
        f.write('\n'.join(data))


def write_json_file(data, path, indent=False):
    with open(path,'w') as f:
        json.dump(data,f, indent=indent)


def write_jsonl_file(data, path, indent=False):
    data = [json.dumps(e, indent=indent) for e in data]
    data = '\n'.join(data)
    with open(path,'w') as f:
        f.write(data)

