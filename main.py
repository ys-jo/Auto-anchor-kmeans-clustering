import os
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.cluster.vq import kmeans
import numpy as np


"""
made by : ys-jo
data : 22.11.09

Reference. https://towardsdatascience.com/training-yolo-select-anchor-boxes-like-this-3226cb8d7f0b

Step 1. Get bounding box sized from the train data
Step 2. Choose a metric to define anchor fitness
    There is a threshold defined as a hyperparameter (called anchor_t, by default 4; sometimes used as 1/anchor_t, which is 0.25). This threshold means that if the anchor box is larger or smaller than the bounding box label no more than 4 times, we assume that it’s a good anchor box.
    We want each bounding box label to be as close as possible to at least one anchor box. And we want it to be close within the threshold (to be no more than 4 times larger or smaller).
    Good fitness is achieved on average, which means that some bounding boxes (probably outliers) may still be far from anchors.
    For each bounding box we select the best anchor, but its fit we calculate from the worse fitting side (hope, it makes sense).
Step 3. Do clustering to get an initial guess for anchors
Step 4. Evolve anchors to improve anchor fitness
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Auto anchor algorithm")
    parser.add_argument('--input', type=str, help='input xml directory path or xml file', required=True)
    parser.add_argument('--img_size', type=int, default = (640,380), help='img size')
    parser.add_argument('--num_box', type=int, default = 16, help='num prior box')
    
    args = parser.parse_args()
    return args

def check_input(arg):
    """
    check dir or file
    ret : list(path)
    """
    data_xml = list()

    if os.path.isfile(arg.input) and arg.input[-3:] == 'xml':
        data_xml.append(arg.input)
        return data_xml
    elif os.path.isdir(arg.input):
        if not arg.input[-1] == '/':
            arg.input = arg.input +'/'
        file_list = os.listdir(arg.input)
        for file in file_list:
            if file[-3:] == 'xml':
                data_xml.append(arg.input + file)
        if data_xml:
            return data_xml
        else:
            raise Exception("Input arg is wrong")
    else:
        raise Exception("Input arg is wrong")

def read_xml(img_size, data_xml):
    final_data = list()
    img_size_width = img_size[0]
    img_size_height = img_size[1]
    print("read xml file")
    for data in tqdm(data_xml):
        tree = ET.parse(data)
        size = tree.find("size")
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        width_ratio = img_size_width/width
        height_ratio = img_size_height/height

        #Letter box (Maintain image resolution ratio)
        if width_ratio >= 1 and height_ratio >= 1:
            if width_ratio > height_ratio:
                height = img_size_height
                width = width * height_ratio
                ratio = height_ratio
            else:
                height = height * width_ratio
                width = img_size_width
                ratio = width_ratio
        elif width_ratio >= 1 and height_ratio <= 1:
            height = img_size_height
            width = width * height_ratio
            ratio = height_ratio
        elif width_ratio <= 1 and height_ratio >= 1:
            height = height * width_ratio
            width = img_size_width
            ratio = width_ratio
        else:
            if width_ratio > height_ratio:
                height = height * width_ratio
                width = img_size_width
                ratio = width_ratio
            else:
                height = img_size_height
                width = width * height_ratio
                ratio = height_ratio

        objects = tree.findall("object")
        for i, obj in enumerate(objects):
            s_data = list()

            bndbox = obj.find("bndbox")
            bnd_width = float(bndbox.find('xmax').text)*ratio - float(bndbox.find('xmin').text)*ratio
            bnd_height = float(bndbox.find('ymax').text)*ratio - float(bndbox.find('ymin').text)*ratio

            s_data.append(round(bnd_width,3))
            s_data.append(round(bnd_height,3))

            final_data.append(s_data)

    """
    findal data structure
    [[width, height].[width, height],[width, height],...]
    """
    return final_data

def get_best_fit(datas, anchor_w, anchor_h, th):
    results = list()
    for data in datas:
        bnd_width = data[0]
        bnd_height = data[1]
        output_list = list()
        for w, h in zip(anchor_w,anchor_h):
            w_ratio = bnd_width/w if bnd_width/w < 1 else w/bnd_width
            h_ratio = bnd_height/h if bnd_height/h < 1 else h/bnd_height
            output_list.append(min(w_ratio,h_ratio))
        if max(output_list) < th:
            results.append(0)
        else:
            results.append(max(output_list))
    results_mean = sum(results)/len(results)
    return round(results_mean,3)

def main(args):
    # Read xml file
    data_xml = check_input(args)
    data = read_xml(args.img_size, data_xml)
    # Define anchor w. h
    anchor_w = [10.0, 16.0, 33.0, 30.0, 62.0, 59.0, 116.0, 156.0, 373.0]
    anchor_h = [13.0, 30.0, 23.0, 61.0, 45.0, 119.0, 90.0, 198.0, 326.0]
    default_th = 0.25
    result = get_best_fit(data, anchor_w, anchor_h, default_th)
    print(f"current prior box (w,h) : { list(zip(anchor_w, anchor_h))}")
    print(f"current score:{result}")
    numpy_data = np.array(data)
    k, dist = kmeans(numpy_data, args.num_box, iter=30)  # points, mean distance
    #sort
    sorted_dict = dict()
    for i, a in enumerate(k):
        area = a[0]*a[1]
        sorted_dict[area] = [a[0], a[1]]
    sorted_list = sorted(sorted_dict.items())
    new_w = list()
    new_h = list()
    for i in sorted_list:
        new_w.append(round(i[1][0], 1))
        new_h.append(round(i[1][1],1))    
    result2 = get_best_fit(data, new_w, new_h, default_th)
    print(f"recommend prior box (w,h) : {list(zip(new_w, new_h))}")
    print(f"recommend score:{result2}")


if __name__ == "__main__":
    
    args = parse_args()
    main(args)
