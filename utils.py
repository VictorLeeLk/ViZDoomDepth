"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: utils.py
"""

import os
import shutil

VGG_MEAN = [103.939, 116.779, 123.68]

def check_file(filename):
    return os.path.isfile(filename)


def check_dir(dirname):
    return os.path.isdir(dirname)


def del_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)


def del_dir_under(dirname):
    os.system('rm -f {}/*'.format(dirname))

def del_file(filename):
    if os.path.isfile(filename):
        os.system('rm ' + filename)


def create_dir(dirname):
    os.mkdir(dirname)


def new_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

    os.mkdir(dirname)


def get_all_files(input_dir, suffix=None):
    files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and (suffix is None or os.path.splitext(file_path)[1] == suffix):
            files.append(file_path)

    return files


def copy_file(from_file, to_file):
    os.system('cp \"{}\" \"{}\"'.format(from_file, to_file))


def get_filename(filepath):
    filepath = filepath.strip()
    while filepath and filepath[-1] == '/':
        filepath = filepath[:-1]

    file_s = filepath.split('/')
    if '.' in file_s[-1]:
        filename = file_s[-1].split('.')[0]
    else:
        filename = file_s[-1]
    return filename
