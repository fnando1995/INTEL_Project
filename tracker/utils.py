import yml

def read_yml(yml_file):
    with open(yml_file) as file:
        data = yml.load(file, Loader=yml.FullLoader)
    return data