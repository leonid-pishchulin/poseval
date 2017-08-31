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

### Evaluation (local)

Evaluation has been tested in Linux and Ubuntu OS. Evaluation supports both challenge tasks, namely per-frame multi-person pose estimation and video-based body pose tracking.
```
$ git clone https://github.com/leonid-pishchulin/poseval.git --recursive
$ cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
$ python evaluate.py --groundTruth=/path/to/annotations/val/ --predictions=/path/to/predictions --evalPoseTracking --evalPoseEstimation
```

### Evaluation (server)

In order to evaluate using evaluation server, zip your directory with json prediction files and submit your results at https://posetrack.net. Shortly you will receive an email containing evaluation results.

For further questions and details, contact Leonid Pishchulin <mailto:leonid.pishchulin@gmail.com>