import os
import json
import  numpy as np

classes = ['0']
# 初始化二维0数组
result_list = np.array(np.zeros([len(classes), len(classes)]))


def read_label_txt(full_label_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = 1280,720
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }

        object_list.append(obj_info)
    return object_list

def read_detelabel_txt(full_label_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = 1280,720
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = (float(array[1]) - float(array[3]) / 2) * image_width
        x_label_max = (float(array[1]) + float(array[3]) / 2) * image_width
        y_label_min = (float(array[2]) - float(array[4]) / 2) * image_high
        y_label_max = (float(array[2]) + float(array[4]) / 2) * image_high
        conf = float(array[5])
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }
        if conf>=0 :
            object_list.append(obj_info)
    return object_list

def read_detlabel_txt(full_label_name):
    fp = open(full_label_name, mode="r")
    lines = fp.readlines()
    image_width, image_high = 1280,720
    object_list = []
    for line in lines:
        array = line.split()
        x_label_min = float(array[1])
        x_label_max = float(array[3])
        y_label_min = float(array[2])
        y_label_max = float(array[4])
        bbox = [round(x_label_min, 2), round(y_label_min, 2), round(x_label_max, 2), round(y_label_max, 2)]
        category = int(array[0])
        obj_info = {
            'category' : category,
            'bbox' : bbox
        }
        object_list.append(obj_info)
    return object_list

# 计算交集面积
def intersection_area(label_box, detect_box):
    
    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box
    if (x_label_max <= x_detect_min or x_detect_max < x_label_min) or ( y_label_max <= y_detect_min or y_detect_max <= y_label_min):
        return 0
    else:
        lens = min(x_label_max, x_detect_max) - max(x_label_min, x_detect_min)
        wide = min(y_label_max, y_detect_max) - max(y_label_min, y_detect_min)
        return lens * wide


# 计算并集面积
def union_area(label_box, detect_box):

    x_label_min, y_label_min, x_label_max, y_label_max = label_box
    x_detect_min, y_detect_min, x_detect_max, y_detect_max = detect_box

    area_label = (x_label_max - x_label_min) * (y_label_max - y_label_min)
    area_detect = (x_detect_max - x_detect_min) * (y_detect_max - y_detect_min)
    inter_area = intersection_area(label_box, detect_box)

    area_union = area_label + area_detect - inter_area

    return area_union


# label 匹配 detect
def label_match_detect(label_list, detect_list):

    #IOU阈值值设置
    iou_threshold = 0.5

    #true_positive_list:存储识别正确的对象，false_positive_list存储识别错误的对象
    true_positive_list = []
    false_positive_list = []
        
        
    for detect in detect_list:
        
        detect_bbox = detect['bbox']
        temp_iou = 0.0
            
        for label in label_list:
            label_bbox = label['bbox']
            i_area = intersection_area(label_bbox, detect_bbox)
            u_area = union_area(label_bbox, detect_bbox)

            iou = i_area / u_area

            #只统计最大IOU的预测对象
            if temp_iou < iou:
                temp_iou = iou
            
        if temp_iou > iou_threshold:
            true_positive_list.append(detect)
        else:
            false_positive_list.append(detect)
    
    return true_positive_list, false_positive_list,label_list



def main():
    count_tp = 0
    count_fp = 0
    count_fn = 0
    count_prec = 0.0
    count_recall = 0.0
    recall = 0
    detect_path = r'/home/zk203/zxd/yolov5-master/runs/detect/exp10/labels/'    # 预测的数据
    label_path = r'/home/zk203/zxd/yolov5-master/data/971_test_data/all/labels/'  # 标注文件路径
    
    all_label = os.listdir(detect_path)

    all_list = []

    for i in range(len(all_label)):
        
        full_detect_path = os.path.join(detect_path, all_label[i])
        # 分离文件名和文件后缀
        label_name, label_extension = os.path.splitext(all_label[i])
        # print (label_name)
        # 拼接标注路径
        full_label_path = os.path.join(label_path, label_name + '.txt')
        full_detect_path = os.path.join(detect_path, label_name + '.txt')
        # 读取标注数据
        if os.path.exists(full_detect_path) == True:
            label_list = read_label_txt(full_label_path)
            # 标注数据匹配detect
            detect_list = read_label_txt(full_detect_path)

            tp_lst, fp_lst,lb_lst = label_match_detect(label_list, detect_list)
            
            obj_info = {
                'label_name' : label_name,
                'tp_lst' : tp_lst,
                'fp_lst' : fp_lst,
                'lb_lst' : lb_lst
            }
            all_list.append(obj_info)
        else:
            myfile = open(full_label_path)
            lines = len(myfile.readlines())
            count_fn += lines



    for lst in all_list:
        tp_lst = lst['tp_lst']
        fp_lst = lst['fp_lst']
        lb_lst = lst['lb_lst']

        count_tp += len(tp_lst)
        count_fp += len(fp_lst)
        if (len(lb_lst)-len(tp_lst)) < 0:
            count_fn += 0
        else:
            count_fn += (len(lb_lst)-len(tp_lst))
        

        prec = count_tp / (count_tp + count_fp)
        
        recall = count_tp / (count_tp + count_fn)

        count_prec += prec
        count_recall += recall

    # print('tp: ',count_tp,'fp: ',count_fp,'fn: ',count_fn)
    # print('pre: ',prec,'recall: ',recall)

    print('tp: ',count_tp,'fn: ',count_fn)
    print('recall: ',recall)
if __name__ == '__main__':
    main()

