import lmdb
import os
import six
import warnings
import PIL
import numpy as np
import cv2


def append_mdb_v2(lmdb1, lmdb2):
    """
    lmdb2 --> lmdb1
    Parameters
    ----------
    lmdb1
    lmdb2

    Returns
    -------

    """
    MAPSIZE = 10 * 1024 * 1024 * 1024 * 1024 * 2

    env1 = lmdb.open(lmdb1, readonly=False, lock=False, readahead=False, meminit=False, map_size=MAPSIZE)
    assert env1, f'Cannot open LMDB dataset from {lmdb1}.'
    env2 = lmdb.open(lmdb2, readonly=True, lock=False, readahead=False, meminit=False)
    assert env2, f'Cannot open LMDB dataset from {lmdb2}.'

    txn1 = env1.begin(write=True)
    txn2 = env2.begin(write=False)

    length1 = int(txn1.get('num-samples'.encode()))
    print("length1: ", length1)
    length2 = int(txn2.get('num-samples'.encode()))
    print("length2: ", length2)

    database1 = txn1.cursor()
    database2 = txn2.cursor()

    count = length1
    key_images = {}
    for (key, value) in database2:
        keyi = str(key, 'utf-8').strip()
        if "image" in keyi:
            count += 1
            imageKey = 'image-%09d'.encode() % count
            key_images[imageKey] = value

    count2 = length1
    key_labels = {}
    for (key, value) in database2:
        keyi = str(key, 'utf-8').strip()
        if "label" in keyi:
            count2 += 1
            labelKey = 'label-%09d'.encode() % count2
            key_labels[labelKey] = value

    cnt = 0
    for k, v in key_images.items():
        txn1.put(k, v)
        if cnt % 1000 == 0:
            txn1.commit()
            txn1 = env1.begin(write=True)
        cnt += 1
    txn1.commit()
    txn1 = env1.begin(write=True)

    cnt2 = 0
    for k, v in key_labels.items():
        txn1.put(k, v)
        if cnt2 % 1000 == 0:
            txn1.commit()
            txn1 = env1.begin(write=True)
        cnt2 += 1
    txn1.commit()
    txn1 = env1.begin(write=True)

    num_new = length1 + length2
    txn1.put('num-samples'.encode(), str(num_new).encode())

    lengthnew = int(txn1.get('num-samples'.encode()))
    assert lengthnew == num_new, "Error: num-samples error!"

    txn1.commit()
    txn1 = env1.begin(write=True)
    print("num-samples: ", int(txn1.get('num-samples'.encode())))

    env1.close()
    env2.close()

    print("OK!")


def read_lmdb(data_path):
    dir_name = os.path.basename(data_path)
    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_{}".format(dir_name, "images_extracted_from_lmdb")
    os.makedirs(save_path, exist_ok=True)

    MAPSIZE = 10 * 1024 * 1024 * 1024 * 1024 * 2

    env1 = lmdb.open(data_path, readonly=False, lock=False, readahead=False, meminit=False, map_size=MAPSIZE)
    assert env1, f'Cannot open LMDB dataset from {data_path}.'

    txn1 = env1.begin(write=True)
    length1 = int(txn1.get('num-samples'.encode()))
    print(length1)

    for idx in range(length1):
        image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
        label = str(txn1.get(label_key.encode()), 'utf-8').strip()  # label

        imgbuf = txn1.get(image_key.encode())  # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        f_dst_path = "{}/{:09d}.jpg".format(save_path, idx + 1)
        cv2.imwrite(f_dst_path, image)

    # database1 = txn1.cursor()
    # for (key, value) in database1:
    #     print(key, value)

    for idx in range(length1):
        image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
        label = str(txn1.get(label_key.encode()), 'utf-8').strip()  # label

        imgbuf = txn1.get(image_key.encode())  # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

        f_dst_path = "{}/{:09d}.jpg".format(save_path, idx + 1)
        cv2.imwrite(f_dst_path, image)

    # database1 = txn1.cursor()
    # for (key, value) in database1:
    #     print(key, value)
    

def change_num_samples(data_path):
    MAPSIZE = 10 * 1024 * 1024 * 1024 * 1024 * 2

    env1 = lmdb.open(data_path, readonly=False, lock=False, readahead=False, meminit=False, map_size=MAPSIZE)
    assert env1, f'Cannot open LMDB dataset from {data_path}.'

    txn1 = env1.begin(write=True)
    length1 = int(txn1.get('num-samples'.encode()))
    print("num-samples: ", length1)

    # database1 = txn1.cursor()
    # txn1.put('num-samples'.encode(), str(6759881).encode())
    # txn1.put('num-samples'.encode(), str(13519733).encode())
    txn1.put('num-samples'.encode(), str(27039469).encode())

    txn1.commit()
    txn1 = env1.begin(write=True)
    print("num-samples: ", int(txn1.get('num-samples'.encode())))

    env1.close()
    print("OK!")




if __name__ == '__main__':
    # lmdb1 <-- lmdb2
    lmdb1 = "/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240927_New/merged_random_selected_0.25_percent_lmdb"
    lmdb2 = "/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240927_New/merged_random_selected_0.25_percent_199693_lmdb"
    append_mdb_v2(lmdb1, lmdb2)

    # read_lmdb(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/experiment/lmdb_test/20241108/test_lmdb")

    # change_num_samples(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/train/txt/20240927_New/merged_lmdb")