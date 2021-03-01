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
from scipy.io import loadmat
from tqdm import tqdm


image_root = '/home/zxiao/data/ROI/roi_test'

# Define colors for visualization

colors = [
    (220,20,60), # Crimson
    (219,112,147), #PaleVioletRed
    (255,215,0), # Gold
    (255,165,0), # Orange
    (255,255,0), # Yellow
    (255,255,224), # LightYellow
    (0,128,0),#Green
    (60,179,113),#SpringGreen
    (0,206,209), #DarkTurquoise
    (0,255,255),#Cyan
    (0,191,255),#DeepSkyBlue
    (70,130,180),#SteelBlue
    (0,0,128), #Navy
    (106,90,205),#SlateBlue
    (138,43,226),#BlueViolet
    (75,0,130),# Indigo
    (148,0,211),#DarkVoilet
    (128,0,128),#Purple
    (186,85,211),#MediumOrchid
    (255,0,255),#Fuchsia
    (238,130,238),#Violet
    (221,160,221),#plum
    (255,20,147),#DeepPink
    (255,192,203),#Pink
    (255,240,245)#LavenderBlush

]


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
    cause = args.cause
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
    for image_file_name in tqdm(vis_info.keys()):
        image_info = vis_info[image_file_name]
        image_dir, image_name = image_file_name.split('/')
        gt_and_pred_dir = os.path.join(vis_info_save_dir.replace('vis_info','vis'),'gt_and_pred',cause+ '_'+ vis_info_save_name.replace('.json',''),image_dir)
        pred_and_score_dir = os.path.join(vis_info_save_dir.replace('vis_info','vis'),'pred_and_score',cause+ '_'+ vis_info_save_name.replace('.json',''),image_dir)
        if not os.path.exists(gt_and_pred_dir):
            os.makedirs(gt_and_pred_dir)
        if not os.path.exists(pred_and_score_dir):
            os.makedirs(pred_and_score_dir)
        gt_and_pred_file = os.path.join(gt_and_pred_dir,image_name.replace('.png','.jpg'))
        pred_and_score_file = os.path.join(pred_and_score_dir,image_name.replace('.png','.jpg'))

        original_image_path = os.path.join(image_root,image_file_name)
        visualizae_gt_and_pred(original_image_path,image_info,gt_and_pred_file)
        visualize_pred_and_score(original_image_path,image_info,pred_and_score_dir)

def visualizae_gt_and_pred(input_image_path,info_dict,save_path):
    image = cv2.imread(input_image_path)
    gt = info_dict['gt']
    score = info_dict['score']
    risk_object_id = np.argmax(score)
    box = info_dict['all_bboxes'][risk_object_id]

    cv2.rectangle(image, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 8)
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3 )
    cv2.imwrite(save_path,image)

def visualize_pred_and_score(input_image_path,info_dict,save_dir):

    image_name = os.path.basename(input_image_path)
    pred_dir = os.path.join(save_dir,'pred')
    score_dir = os.path.join(save_dir,'score')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)

    #colors = loadmat('/home/zxiao/project/semantic-segmentation-pytorch/data/color150.mat')['colors']
    image = cv2.imread(input_image_path)
    num_box = info_dict['num_box'] -1 
    scores = info_dict['score'][1:]
    all_boxes = info_dict['all_bboxes'][1:]
    leftmost = [x[0] for x in all_boxes]
    sorted_scores = [x for _, x in sorted(zip(leftmost,scores), key=lambda pair: pair[0])]
    sorted_boxes = [x for _, x in sorted(zip(leftmost,all_boxes), key=lambda pair: pair[0])]
    confidence_go = info_dict['confidence_go']
    # save pred
    for i in range(num_box):
        box = sorted_boxes[i]
        r,g,b = colors[i]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (int(b), int(g), int(r)), 2 )
    cv2.imwrite(os.path.join(pred_dir,image_name.replace('.png','.jpg')),image)

    # save score
    color_tuples = [(float(r/255),float(g/255),float(b/255)) for r,g,b in colors]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(range(len(sorted_scores)), sorted_scores,color = color_tuples[:num_box])
    ax.axhline(confidence_go,0,len(sorted_scores),color = 'k')
    ax.set_ylim([0,1.0])
    ax.set_xticks([])
    ax.set_title('Risk Score')
    ax.set_xlabel('Object')
    fig.savefig(os.path.join(score_dir,image_name.replace('.png','.jpg')))
    plt.close(fig)
if __name__ == '__main__':
    main()