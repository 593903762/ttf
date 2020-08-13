import os
# import cv2
import sys
import pandas as pd
from PIL import Image, ImageDraw,ImageFont
import json
# sys.append('')

def plt_box(img_cat):
    base_path = '../data/WiderPerson'
    path = os.path.join(base_path, img_cat+'.txt')
    with open(path, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]
    
    save_train_img_dir = os.path.join(base_path, img_cat+'_image_with_bbox')
    if not os.path.exists(save_train_img_dir):
        os.makedirs(save_train_img_dir)

    for img_id in img_ids: # '000040'
        img_path = os.path.join(base_path,'Images',img_id+'.jpg')
        if not os.path.exists(img_path):
            print(img_path)
            continue
        img = cv2.imread(img_path)

        im_h = img.shape[0]
        im_w = img.shape[1]
 
        label_path = img_path.replace('Images', 'Annotations') + '.txt'
 
        with open(label_path) as file:
            line = file.readline()
            count = int(line.split('\n')[0]) # 里面行人个数
            line = file.readline()
            while line:
                cls = int(line.split(' ')[0])
                # < class_label =1: pedestrians > 行人
                # < class_label =2: riders >      骑车的
                # < class_label =3: partially-visible persons > 遮挡的部分行人
                # < class_label =4: ignore regions > 一些假人，比如图画上的人
                # < class_label =5: crowd > 拥挤人群，直接大框覆盖了
                if cls == 1 or cls == 2 or cls == 3:
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    # cv2的颜色是BGR
                    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                else:
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                line = file.readline()
        # cv2.imshow('result', img)
        cv2.imwrite(os.path.join(save_train_img_dir, img_id+'.jpg'), img)
        # cv2.waitKey(0)

 

def draw_rectangle_img_from_pd(one_img_info_pd, img_path, save_img_path):
    '''
    one_img_info_pd:pd形式，保存了所有的一张图片中的所有相关信息
    img_path:打开的img路径
    save_img_path:画好图之后img的保存路径
    '''
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
    # font = ImageFont.truetype("arial.ttf", 20)
    # width = bb[2]
    # height = bb[3]
    # bb[2] = width+bb[0]
    # bb[3] = height+bb[1]
    for i in one_img_info_pd.index:
        bb = one_img_info_pd.loc[i, 'bbox']
        xmin = bb[0]
        ymin = bb[1]
        width = bb[2]
        height = bb[3]
        if one_img_info_pd.loc[i, 'category_id'] == 1:
            draw.rectangle((xmin, ymin, xmin+width, ymin+height), outline='#00ff00', width=3)
        if one_img_info_pd.loc[i, 'category_id'] == 2:
            draw.rectangle((xmin, ymin, xmin+width, ymin+height), outline='#0000ff', width=3)
    img.save(save_img_path)
    return

def draw_rectangle_img(img_dir_path, result_json_path, score, save_img_dir):
    '''
    widerpserson中infer出来的bbox是按照(xmin, ymin, width, height)这个顺序排列
    而PIL.rectangle画图正好是按照(xmin, ymin, xmax, ymax)进行画图
    从widerperson的val文件夹中读取图片，然后从这个json文件中读取对应的bbox进行画框，最后保存
    '''
    
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    # step1: 读取json文件为pd类型
    def get_dict(path):
        with open(path,'r') as f:
            train_2017 = f.read()
            train_2017 = json.loads(train_2017)
        return train_2017
    widerperson_infer_path  = r'/home/qinyuanze/code/ttf2/ttfnet/wider_person_val_infer_result.json'
    widerperson_infer = get_dict(widerperson_infer_path)
    pd_widerperson_infer = pd.DataFrame(widerperson_infer)


    # step2: 读取图片,并进行画图
    img_list = os.listdir(img_dir_path)
    for img_path in img_list:
        img_id = int(img_path.split(r'.jpg')[0].strip('0'))
        img_info = pd_widerperson_infer[(pd_widerperson_infer.image_id==img_id) \
         & (pd_widerperson_infer.score >= score)]
        if not img_info.empty:
            save_img_path = os.path.join(save_img_dir, img_path)
            draw_rectangle_img_from_pd(img_info, os.path.join(img_dir_path,img_path), save_img_path)

    

if __name__ == '__main__':
    # plt_box('val')
    draw_rectangle_img(img_dir_path=r'/home/qinyuanze/code/ttf2/ttfnet/data/coco/val2017',
                     result_json_path=r'/home/qinyuanze/code/ttf2/ttfnet/wider_person_val_infer_result.json', 
                     score=0.4,
                     save_img_dir = r'/home/qinyuanze/code/ttf2/ttfnet/data/WiderPerson/val_infer')
