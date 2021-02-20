import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../../')
import cv2
import json
import argparse
import pdb



image_root = '/home/zxiao/data/ROI/roi_test'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='./snapshots/crossing_vehicle/2020-2-13_175803_result_w_dataAug/inputs-camera-epoch-1.pth',
                        type=str)
    parser.add_argument('--cause',
                        default='crossing_vehicle',
                        type=str)
    args = parser.parse_args()
    dir_name = os.path.dirname(args.model)
    model_basename = os.path.basename(args.model)
    vis_info_save_dir = os.path.join(dir_name,'vis_info')
    vis_info_save_name = model_basename.replace('.pth','.json')
    vis_info_file_path = os.path.join(vis_info_save_dir,args.cause+ '_'+ vis_info_save_name)
    print(args.model)
    print(args.cause)
    print(vis_info_file_path)
    if not os.path.exists(vis_info_file_path):
        print("Vis_info file doesn't exist. Please run risk object identification first")
    else:
        print("Loading vis_info from {} ...".format(vis_info_file_path))
    with open(vis_info_file_path,'r') as f:
        vis_info = json.load(f)
    for image_file_name in vis_info.keys():
        image_info = vis_info[image_file_name]
        image_dir, image_name = image_file_name.split('/')
        gt_and_pred_dir = os.path.join(vis_info_save_dir.replace('vis_info','vis'),'gt_and_pred',image_dir)
        pred_and_score_dir = os.path.join(vis_info_save_dir.replace('vis_info','vis'),'pred_and_score',image_dir)
        if not os.path.exists(gt_and_pred_dir):
            os.makedirs(gt_and_pred_dir)
        if not os.path.exists(pred_and_score_dir):
            os.makedirs(pred_and_score_dir)
        gt_and_pred_file = os.path.join(gt_and_pred_dir,image_name.replace('.png','.jpg'))
        pred_and_score_file = os.path.join(pred_and_score_dir,image_name.replace('.png','.jpg'))

        print(gt_and_pred_file)
        print(pred_and_score_file)
        original_image_path = os.path.join(image_root,image_file_name)
        visualizae_gt_and_pred(original_image_path,image_info,gt_and_pred_file)

def visualizae_gt_and_pred(input_image_path,info_dict,save_path):
    image = cv2.imread(input_image_path)
    gt = info_dict['gt']
    score = info_dict['score']
    risk_object_id = np.argmax(score)
    box = info_dict['all_bboxes'][risk_object_id]

    cv2.rectangle(image, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 8)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3 )
    cv2.imwrite(save_path,image)
if __name__ == '__main__':
    main()