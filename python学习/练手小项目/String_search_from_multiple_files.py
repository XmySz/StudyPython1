import os

string = "i like you"
path = r"E:\pythonProject\python学习"


def find_file(path):
    find_flag = False
    os.chdir(path)
    files = os.listdir()

    for file in files:
        abs_path = os.path.abspath(file)
        if os.path.isdir(abs_path):
            find_file(file)
        if os.path.isfile(abs_path):
            with open(file, 'r') as f:
                if string in f.read():
                    find_flag = True
                    print(f"{string} found in  {os.path.abspath(file)}")
                    return True

    if find_flag:
        print(f"{string} not found!")
        return False


find_file(path)
