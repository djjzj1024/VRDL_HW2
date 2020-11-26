import json
import os
import mmcv
import numpy as np

def create_digitStruct_nobbox(test_folder, out_file):
    """
    Create an initial annotation file for data, where contains only filename
    but no ground truth for each images.

    :param test_folder: Where test data placed.
    :param out_file: Output file name.
    """
    data_info = os.listdir(test_folder) # May not be ordered
    data_info.sort(key=lambda name: int(name[:-4])) # Order dir by filename
    
    ann = []
    for name in mmcv.track_iter_progress(data_info):
        if not name.endswith('.png'):
            continue
        ann.append(dict(filename=name, boxes=[]))
    
    with open(out_file, 'w') as f:
        json.dump(ann, f)

def convert_digitStruct_to_COCO(ann_file, out_file, image_prefix, categories):
    """
    Convert digitStruct.json to COCO format.

    :param ann_file: Annotation file, could be filename or its content, format
        in digitStruct.json way.
    :param out_file: Output file name.
    :param image_prefix: Prefix of image, the directory where images at.
    :param categories: Categories with following format:
        list of dictionary with keys 'id' and 'name', such as
        [{'id': id, 'name': name}, ]
    """
    if isinstance(ann_file, str):
        data_infos = mmcv.load(ann_file)
    elif isinstance(ann_file, list):
        data_infos = ann_file
    images = []
    annotations = []
    obj_count = 0
    for idx, v in enumerate(mmcv.track_iter_progress(data_infos)):
        filename = v['filename']
        img_path = os.path.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=filename,
            height=height,
            width=width))

        for box in v['boxes']:
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=int(box['label']),
                bbox=[box['left'], box['top'], box['width'], box['height']],
                area=box['width'] * box['height'],
                iscrowd=0)
            
            annotations.append(data_anno)
            obj_count += 1
        
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    mmcv.dump(coco_format_json, out_file)

def split_train_val_to_annotaion(ann_file, train_out, val_out,
                               image_prefix, categories, val_rate=0.2):
    """
    Use the input digitStruct.json to create COCO format annotation file for
    training and validation.

    :param ann_file: Annotation file name, a digitStruct.json for all data.
    :param train_out: Output file name for training annotation.
    :param val_out: Output file name for validation annotation.
    :param image_prefix: Prefix of image, the directory where images at.
    :param categories: Categories with following format:
        list of dictionary with keys 'id' and 'name', such as
        [{'id': id, 'name': name}, ]
    :param val_rate: Percentage of validation set in whold dataset.
    """
    data_info = mmcv.load(ann_file)
    total = len(data_info)
    choice = np.random.choice(total, size=int(total*val_rate), replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[choice] = True
    
    data_info = np.asarray(data_info)
    train_data = data_info[~mask].tolist()
    val_data = data_info[mask].tolist()

    convert_digitStruct_to_COCO(train_data, train_out, image_prefix, categories)
    convert_digitStruct_to_COCO(val_data, val_out, image_prefix, categories)
