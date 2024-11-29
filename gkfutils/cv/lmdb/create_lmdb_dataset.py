""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True, map_size=5073741824):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=map_size)
    cache = {}
    cnt = 1

    datalist = open(gtFile, 'r', encoding='utf-8').read().strip().split('\n')

    print(len(datalist))
    for i, sample in tqdm(enumerate(datalist)):
        try:
            imagePath, label = sample.split('\t')
            if len(label) < 51:
                imagePath = os.path.join(inputPath, imagePath)

                # # only use alphanumeric data
                # if re.search('[^a-zA-Z0-9]', label):
                #     continue

                if not os.path.exists(imagePath):
                    print('%s does not exist' % imagePath)
                    continue
                with open(imagePath, 'rb') as f:
                    imageBin = f.read()
                if checkValid:
                    try:
                        if not checkImageIsValid(imageBin):
                            print('%s is not a valid image' % imagePath)
                            continue
                    except:
                        print('error occured', i)
                        with open(outputPath + '/error_image_log.txt', 'a') as log:
                            log.write('%s-th image data occured error\n' % str(i))
                        continue

                imageKey = 'image-%09d'.encode() % cnt
                labelKey = 'label-%09d'.encode() % cnt
                cache[imageKey] = imageBin
                cache[labelKey] = label.strip().encode()

                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d / %d' % (cnt, i))
                cnt += 1
        except Exception as e:
            print(sample, e)

    i = cnt - 1
    cache['num-samples'.encode()] = str(i).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % i)


def createDataset_v2(data_path, checkValid=True, map_size=5073741824, alpha=None):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    base_name = os.path.basename(data_path)
    fname = os.path.splitext(base_name)[0]

    save_path = os.path.abspath(os.path.join(data_path, "..")) + "/{}_lmdb".format(fname)
    os.makedirs(save_path, exist_ok=True)
    print("save_path: ", save_path)

    env = lmdb.open(save_path, map_size=map_size)
    cache = {}
    cnt = 1

    fr = open(data_path, 'r', encoding='utf-8')
    datalist = fr.readlines()
    fr.close()
    len_d = len(datalist)
    print("len_d: ", len_d)

    for i, sample in tqdm(enumerate(datalist)):
        try:
            imagePath = sample.split(' ')[0]
            label = sample.split(' ')[1].strip()

            if not os.path.exists(imagePath):
                continue

            if len(label) > 15:
                continue

            # if len(label) < 51:
            # imagePath = os.path.join(inputPath, imagePath)

            # # only use alphanumeric data
            # if re.search('[^a-zA-Z0-9]', label):
            #     continue

            num_ = 0
            for l in label:
                if l not in alpha:
                    num_ += 1
            if num_ > 0:
                continue

            # num_special = 0
            # for li, l in enumerate(label.strip()):
            #     if li == 0 and l in "°²³":
            #         num_special += 1
            # if num_special > 0:
            #     continue

            # num_special2 = 0
            # for li, l in enumerate(label.strip()):
            #     if l in "°²³":
            #         num_special2 += 1
            # if num_special2 == len(label.strip()):
            #     continue

            if label == "":
                continue

            if not os.path.exists(imagePath):
                print('%s does not exist' % imagePath)
                continue

            with open(imagePath, 'rb') as f:
                imageBin = f.read()

            if checkValid:
                try:
                    if not checkImageIsValid(imageBin):
                        print('%s is not a valid image' % imagePath)
                        continue
                except:
                    print('error occured', i)
                    with open(save_path + '/error_image_log.txt', 'a') as log:
                        log.write('%s-th image data occured error\n' % str(i))
                    continue

            imageKey = 'image-%09d'.encode() % cnt
            labelKey = 'label-%09d'.encode() % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.strip().encode()

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, len_d))
            cnt += 1

        except Exception as e:
            print(sample, e)

    i = cnt - 1
    cache['num-samples'.encode()] = str(i).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % i)


def read_ocr_lables(lbl_path):
    CH_SIM_CHARS = ' '
    ch_sim_chars = open(lbl_path, "r", encoding="utf-8")
    lines = ch_sim_chars.readlines()
    for l in lines:
        CH_SIM_CHARS += l.strip()
    alpha = CH_SIM_CHARS  # len = 1 + 6867 = 6868
    return alpha


if __name__ == '__main__':
    MAPSIZE = 10 * 1024 * 1024 * 1024 * 1024 * 2

    # alpha = ' ' + '0123456789' + '.:/\\-' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # createDataset_v2(data_path="/home/disk/disk7/data/000.OpenDatasets/OCR/Merged_train_test/Merged_train.txt", checkValid=True, map_size=MAPSIZE, alpha=alpha)

    alpha_21160 = read_ocr_lables("/home/wujiahu/GraceKafuu/GraceKafuu_v1.0.0/Python/CV_v1.0.0/OCR/PyTorchOCR/Rec/CRNN/CRNN_PyTorch_2024.08.02/words/chinese_chars_v1_21159.txt")
    createDataset_v2(data_path="/home/disk/disk7/wujiahu/data/000.Data/ocr/chn/ChineseOCR/data/v2/horizontal/test/txt/merged.txt", checkValid=True, map_size=MAPSIZE, alpha=alpha_21160)


    
