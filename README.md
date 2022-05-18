# The Something-Else Annotations
This repository provides instructions regarding the annotations used in the paper: 'Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks' (https://arxiv.org/abs/1912.09930).
We collected annotations for 180049 videos from the Something-Something Dataset (https://20bn.com/datasets/something-something), that include per frame bounding box annotation for each object and hand in the human-object interaction in the video.

The file containing annotations can be downloaded from:

https://drive.google.com/open?id=1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ in four parts,
it containes a dictionary mapping each video id, the name of the video file to the list of per-frame annotations. The annotations assume that the frame rate of the videos is 12.
An example of per-frame annotation is shown below, the names and number of "something's" in the frame correspond to the fields
'gt_placeholders' and 'nr_instances', the frame path is given in the field 'name', 'labels' is a list of object's and hand's bounding boxes and names.

```
   [
    {'gt_placeholders': ['pillow'],
     'labels': [{'box2d': {'x1': 97.64950730138266,
                          'x2': 427,
                          'y1': 11.889318166856967,
                          'y2': 239.92858832368972},
                          'category': 'pillow',
                          'standard_category': '0000'}},
                {'box2d': {'x1': 210.1160330781122,
                          'x2': 345.4329005999551,
                          'y1': 78.65516045335991,
                          'y2': 209.68758889799403},
                          'category': 'hand',
                          'standard_category': 'hand'}}],
     'name': '2/0001.jpg',
     'nr_instances': 2}, 
     {...},
     ...
     {...},
     ]
```
The annotations for example videos are a small subset of the annotation file, and can be found in `annotations.json`.

# Project website: 
https://joaanna.github.io/something_else/

# Citation
If you use our annotations in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{CVPR2020_SomethingElse,
  title={Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks},
  author={Materzynska, Joanna and Xiao, Tete and Herzig, Roei and Xu, Huijuan and Wang, Xiaolong and Darrell, Trevor},
  booktitle = {CVPR},
  year={2020}
}

@inproceedings{goyal2017something,
  title={The" Something Something" Video Database for Learning and Evaluating Visual Common Sense.},
  author={Goyal, Raghav and Kahou, Samira Ebrahimi and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={ICCV},
  volume={1},
  number={4},
  pages={5},
  year={2017}
}
```


# Dataset splits
The compositional, compositional shuffle, one-class compositional and few-shot splits of the Something Something v2 Dataset are available in the folder `dataset_splits`. 

# Visualization of the ground-truth bounding boxes
The folder `videos` contains example videos from the dataset and selected annotations file (full file available on google drive). To visualize videos with annotated bounding boxes run:

```python annotate_videos.py```

The annotated videos will be saved in the `annotated_videos` folder.

# Visualization of the detected bounding boxes 

![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/10015.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/130153.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/154439.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/174270.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/17628.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/21037.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/24719.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/31061.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/31156.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/35176.gif)

# Training
To train the models from our paper run:
```python train.py --model coord_latent_nl --num_frames 16 --logname experiment_name --batch_size 12 
                   --coord_feature_dim 256 --root_frames /path/to/frames 
                   --json_data_train dataset_splits/compositional/train.json 
                   --json_data_val dataset_splits/compositional/validation.json 
                   --json_file_labels dataset_splits/compositional/labels.json
                   --tracked_boxes /path/to/bounding_box_annotations.json
```

Place the data in the folder /path/to/frames each video bursted into frames in a separate folder. The ground-truth box annotations 
can be found in the google drive in parts and have to be concatenated in a single json file.

The models that are using appearance features are initialized with I3D network pre-trained on Kinetics, the checkpoint can be found in the google drive and should be placed in 'model/pretrained_weights/kinetics-res50.pth'.

We also provide some checkpoints to the trained models. To evaluate a model use the same script as for training with a flag `--args.evaluate` and path to the checkpoint `--args.resume /path/to/checkpoint`'

# Acknowledgments
We used parts of code from following repositories:
https://github.com/facebookresearch/SlowFast
https://github.com/TwentyBN/something-something-v2-baseline
