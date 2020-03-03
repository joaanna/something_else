import argparse
import json
import os
import itertools
import glob
from PIL import Image
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-vid_path', default='videos/', type=str,
                    help='Path to the directory with videos')
parser.add_argument('-out_vid_path', default='annotated_videos/', type=str,
                    help='Path to the directory where the annotated videos are saved')
parser.add_argument('--annot_path', default='annotations.json', type=str,
                    help='Path to the box annotation file')
parser.add_argument('--annotate_objects', default=True, type=bool,
                    help='Flag indicating whether to annotate the bounding boxes with names of objects')

COLORS = [(255, 64, 64), (0, 0, 255), (127, 255, 0), (255, 97, 3), (220, 20, 60),
          (255, 185, 15), (255, 20, 147), (255, 105, 180), (60, 179, 113)]


def annotate_frame(meta, color_map, args):
    frame_path = os.path.join(args.vid_path, meta['name'])
    image = Image.open(frame_path)
    image = np.array(image, dtype=np.uint8)
    for i in meta['labels']:
        x1, x2, y1, y2 = i['box2d']['x1'], i['box2d']['x2'], i['box2d']['y1'], i['box2d']['y2']
        image = cv2.rectangle(
            image, tuple([int(x1), int(y1)]), tuple([int(x2), int(y2)]), color_map[i['category']], 4)
        if args.annotate_objects:
            cv2.putText(image, i['category'], (int(x1)+5, int(y1)+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
        im = Image.fromarray(image)
        im.save(os.path.join(args.out_vid_path, meta['name']))

def get_colormap(meta):
    all_objects = [[j['category'] for j in i['labels']] for i in meta]
    all_objects = np.unique(list(itertools.chain.from_iterable(all_objects)))
    color_map = {all_objects[i]: COLORS[i] for i in range(len(all_objects))}
    return color_map



def annotate_video(video_path, annotations, args):
    vid_id = video_path.split('/')[-1]
    try:
        meta = annotations[vid_id]
    except:
        print('Annotations for the video {} not found, skipping!'.format(vid_id))
        return
    os.makedirs(os.path.join(args.out_vid_path, vid_id), exist_ok=True)
    color_map = get_colormap(meta)
    for meta_frame in meta:
        annotate_frame(meta_frame, color_map, args)


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.annot_path, 'r') as f:
        annotations = json.load(f)

    os.makedirs(args.out_vid_path, exist_ok=True)

    video_paths = glob.glob(args.vid_path + '/*')
    for video_path in video_paths:
        annotate_video(video_path, annotations, args)
