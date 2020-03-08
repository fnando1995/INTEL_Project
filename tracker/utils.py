import yaml

def read_yml(yml_file):
    with open(yml_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data