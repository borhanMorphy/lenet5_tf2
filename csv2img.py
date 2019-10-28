import os
import numpy as np
import pandas as pd
import cv2
import multiprocessing
import argparse

def get_arguments():
    args = argparse.ArgumentParser("CSV to image converter")
    args.add_argument("--worker_count","-wc",
        default=multiprocessing.cpu_count(),
        help="worker count for the script", type=int)

    return args.parse_args()

def row2img(args):
    file_path,row = args
    img = np.array([row[k] for k in row.keys()],dtype=np.uint8).reshape(28,28)
    cv2.imwrite(file_path,img)

def iterate_over_df(csv_file_name, p, batch_size):
    folder_path = ".".join(csv_file_name.split(".")[:-1])

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    # load dataframe
    df = pd.read_csv(csv_file_name)
    batch = []
    for index,row in df.iterrows():
        
        row = dict(row)
        if "label" in row:
            label = row['label']
            del row['label']
            file_path = os.path.join(folder_path,"{}_{:06d}.jpg".format(label,index))
        else:
            file_path = os.path.join(folder_path,"{:06d}.jpg".format(index))
        
        batch.append((file_path,row))
        
        if len(batch) == batch_size:
            p.map(row2img,batch)
            batch = []
    p.map(row2img,batch)

def convert_csv_to_img(csv_folder_path, worker_count):
    csv_train_path = os.path.join(csv_folder_path,"train.csv")
    csv_test_path = os.path.join(csv_folder_path,"test.csv")

    p = multiprocessing.Pool(worker_count)
    iterate_over_df(csv_train_path,p,worker_count*2)
    iterate_over_df(csv_test_path,p,worker_count*2)