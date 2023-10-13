import os, re, random
import csv

label_index = {'Swiping Left':0,'Swiping Right':1,'Swiping Down':2,'Swiping Up':3,
               'Pushing Hand Away':4,'Pulling Hand In':5,'Sliding Two Fingers Left':6,
               'Sliding Two Fingers Right':7,'Sliding Two Fingers Down':8,
               'Sliding Two Fingers Up':9,'Pushing Two Fingers Away':10,
               'Pulling Two Fingers In':11,'Rolling Hand Forward':12,
               'Rolling Hand Backward':13,'Turning Hand Clockwise':14,
               'Turning Hand Counterclockwise':15,'Zooming In With Full Hand':16,
               'Zooming Out With Full Hand':17,'Zooming In With Two Fingers':18,
               'Zooming Out With Two Fingers':19,'Thumb Up':20,'Thumb Down':21,
               'Shaking Hand':22,'Stop Sign':23,'Drumming Fingers':24,'No gesture':25,
               'Doing other things':26,}

def get_csv_file(csv_path):
    label_data = {}
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            row = row[0].split(';')
            label_data[row[0]] = label_index[row[1]]
    return label_data, sorted(label_data, key=str2int)

def get_file_name(file_path):
    for root, dirs, _ in os.walk(file_path):
        for i, dir_name in enumerate(dirs):
            dirs[i] = os.path.join(root, dir_name)
        return sorted(dirs, key=str2int)

def get_frame(video_path):
    imagelist = []
    for root, _, filenames in os.walk(video_path):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(root, filename))
        return sorted(imagelist, key=str2int)

def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index

def str2int(v_str):
    def tryint(s):
        try:
            return int(s)
        except ValueError:
            return s
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

