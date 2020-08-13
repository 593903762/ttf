import os
import json
import cv2

class get_instance(object):
    def __init__(self, img_dir, anno_txt_dir, save_json_path, start_id = 1):
        self.img_dir = img_dir
        self.anno_txt_dir = anno_txt_dir
        self.save_json_path = save_json_path
        self.start_id = start_id
        
    def get_img_images(self):
        list_txt = os.listdir(self.img_dir)
        images_info = []
        for image_name in list_txt:
            # print(image_name)
            img_path = os.path.join(self.img_dir, image_name)
            img_info = cv2.imread(img_path).shape
            img_width,img_height = img_info[1], img_info[0]
            img_id = int(image_name.split(r'.jpg')[0].strip('0'))
            one_image = {'file_name':image_name,
                        'height':img_height,
                        'width':img_width,
                        'id':img_id}
            images_info.append(one_image)
        return images_info

    def get_img_annotations(self):
        images_annotations = []
        bbox_id = self.start_id
        def get_one_image_one_segmentation(bbox):
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]+xmin
            ymax = bbox[3]+ymin
            annotations = {'segmentation':[[
                xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]],
                'area':(xmax-xmin)*(ymax-ymin),
                'iscrowd':0,
                'bbox':bbox}
            return annotations
            
        person_cat = [1,2,3]
        image_list = os.listdir(self.img_dir)
        for ind,image_name in enumerate(image_list):
            from_txt_path = os.path.join(self.anno_txt_dir, image_name+'.txt')
            if not os.path.exists(from_txt_path):
                print(from_txt_path)
                break
            with open(from_txt_path, 'r') as f:
                bbox_num = f.readline().split('\n')[0]
                info = f.readline().split('\n')[0]
                while info:
                    xmin = float(info.split(' ')[1])
                    ymin = float(info.split(' ')[2])
                    xmax = float(info.split(' ')[3])
                    ymax = float(info.split(' ')[4])
                    bbox_cat = int(info.split(' ')[0])
                    bb = [xmin, ymin, xmax-xmin, ymax-ymin]
                    annotation = get_one_image_one_segmentation(bb)
                    if bbox_cat in person_cat:
                        annotation['category_id'] = 1
                    else:
                        annotation['category_id'] = 2
                    annotation['id']=bbox_id
                    annotation['image_id'] = int(image_name.split(r'.jpg')[0].strip('0'))
                    bbox_id += 1
                    images_annotations.append(annotation)
                    info = f.readline().split('\n')[0]
        return images_annotations

    def write_instances_json(self):
        images_image = self.get_img_images()
        images_annotations = self.get_img_annotations()
        categories = [{'supercategory':'person', 'id': 1, 'name':'person'},
              {'supercategory':'valid', 'id': 2, 'name':'things'}]
        
        train_write = {'images':images_image,
                   'annotations':images_annotations,
                   'categories':categories}
                
        with open(self.save_json_path,'w') as f:
            f.write(json.dumps(train_write))


if __name__ == "__main__":
    annotation_dir = r'/home/qinyuanze/code/ttf2/ttfnet/data/WiderPerson/Annotations'

    train_img_dir = r'/home/qinyuanze/code/ttf2/ttfnet/data/WiderPerson/train'
    train_save_json_path = r'/home/qinyuanze/code/ttf2/ttfnet/data/coco/annotations/instances_train2017.json'
    get_train_instance = get_instance(img_dir=train_img_dir,
                                    anno_txt_dir=annotation_dir,
                                    save_json_path=train_save_json_path)
    get_train_instance.write_instances_json()

    val_img_dir = r'/home/qinyuanze/code/ttf2/ttfnet/data/WiderPerson/val'
    val_save_json_path = r'/home/qinyuanze/code/ttf2/ttfnet/data/coco/annotations/instances_val2017.json'
    get_val_instance = get_instance(img_dir=val_img_dir,
                                    anno_txt_dir=annotation_dir,
                                    save_json_path=val_save_json_path,
                                    start_id=28424)
    get_val_instance.write_instances_json()




def draw_rectangle_img(img_path, bb, info_txt_path):
    '''
    widerpserson中的bbox是按照(xmin, ymin, xmax, ymax)这个顺序排列
    而PIL.rectangle画图正好是按照(xmin, ymin, xmax, ymax)进行画图
    但是coco数据集是按照annotation数据标注格式分别为(x,y,width,height)
    '''
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"/usr/share/fonts/truetype/ubuntu/UbuntuMono-RI.ttf", size=20)
    # font = ImageFont.truetype("arial.ttf", 20)
    # width = bb[2]
    # height = bb[3]
    # bb[2] = width+bb[0]
    # bb[3] = height+bb[1]
    with open(info_txt_path, 'r') as f:
        num = f.readline()
        bbs = []
        info = f.readline().split('\n')[0]
        while info:
            xmin = float(info.split(' ')[1])
            ymin = float(info.split(' ')[2])
            xmax = float(info.split(' ')[3])
            ymax = float(info.split(' ')[4])
            bb = [xmin, ymin, xmax, ymax]
            bbs.append(bb)
            info = f.readline().split('\n')[0]
    for bb in bbs:
        print(bb)
        draw.rectangle(tuple(bb), outline='#ff0000', width=3)
        # draw.text((bb[0], bb[1]), '', font=font, fill="#ff0000")
    img_file_name = os.path.split(img_path)[-1]
    
    save_path = os.path.join('./', img_file_name)
    print(save_path)
    img.save(save_path)