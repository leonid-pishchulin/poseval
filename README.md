Evaluation of Multi-Person Pose Estimation and Tracking on PoseTrack Dataset
=====================

Created by Leonid Pishchulin

### Introduction

This readme provides step-by-step instructions how to evaluate predictions on val set of [PoseTrack Dataset](https://posetrack.net) locally on your machine. For evaluation on val and test set using evaluation server, see below.

### Prerequisites

- numpy>=1.12.1
- pandas>=0.19.2
- scipy>=0.19.0

### Install
```
$ git clone https://github.com/leonid-pishchulin/poseval.git --recursive
$ cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
```
### Data preparation

Evaluation requires ground truth (GT) annotations available at [PoseTrack](https://posetrack.net) and  your method's predictions. Both GT annotations and your predictions must be saved in json format. Following GT annotations, predictions must be stored per sequence using the same structure as GT annotations, and have the same filename as GT annotations.
We provide a possibility to convert a Matlab structure into json format.
```
$ cd poseval/matlab
$ matlab -nodisplay -nodesktop -r "mat2json('/path/to/dir/with/mat/files/'); quit"
```

### Metrics

This code allows to perform evaluation of per-frame multi-person pose estimation and evaluation of video-based multi-person pose tracking.

#### Per-frame multi-person pose estimation

Average Precision (AP) metric is used for evaluation of per-frame multi-person pose estimation. Our implementation follows the measure proposed in [1] and requires predicted body poses with body joint detection scores as input. First, multiple body pose predictions are greedily assigned to the ground truth (GT) based on the highest PCKh [3]. Only single pose can be assigned to GT. Unassigned predictions are counted as false positives. Finally, part detection score is used to compute AP for each body part. Mean AP over all body parts is reported as well.

#### Video-based pose tracking

Multiple Object Tracking (MOT) metrics [2] are used for evaluation of video-based pose tracking. Our implementation builds on the MOT evaluation code [4] and requires predicted body poses with tracklet IDs as input. First, for each frame, for each body joint class, distances between predicted locations and GT locations are computed. Then, predicted tracklet IDs and GT tracklet IDs are taken into account and all (prediction, GT) pairs with distances not exceeding PCKh [3] threshold are considered during global matching of predicted tracklets to GT tracklets for each particular body joint. Global matching minimizes the total assignment distance. Finally, Multiple Object Tracker Accuracy (MOTA), Multiple Object Tracker Precision (MOTP), Precision and Recall metrics are computed. We report MOTA metric for each body joint class and average over all body joints, while for MOTP, Precision and Recall we report averages only.

### Evaluation (local)

Evaluation code has been tested in Linux and Ubuntu OS. Evaluation takes as input path to directory with GT annotations and path to directory with predictions. See "Data preparation" for details on prediction format. 

```
$ git clone https://github.com/leonid-pishchulin/poseval.git --recursive
$ cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
$ python evaluate.py --groundTruth=/path/to/annotations/val/ --predictions=/path/to/predictions --evalPoseTracking --evalPoseEstimation
```

Evaluation of multi-person pose estimation requires joint detection scores, while evaluation of pose tracking requires predicted tracklet IDs per pose.

### Evaluation (server)

In order to evaluate using evaluation server, zip your directory with json prediction files and submit your results at https://posetrack.net. Shortly you will receive an email containing evaluation results.

For further questions and details, contact PoseTrack Team <mailto:admin@posetrack.net>