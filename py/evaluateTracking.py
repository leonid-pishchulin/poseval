import numpy as np
import json
import os
import sys

import eval_helpers
from eval_helpers import Joint
sys.path.append('../../py-motmetrics/')
import motmetrics as mm


def computeMetrics(gtFramesAll, motAll):

    assert(len(gtFramesAll) == len(motAll))

    nJoints = Joint().count
    seqidxs = []
    for imgidx in range(len(gtFramesAll)):
        seqidxs += [gtFramesAll[imgidx]["seq_id"]]
    seqidxs = np.array(seqidxs)

    seqidxsUniq = np.unique(seqidxs)

    mh = mm.metrics.create()

    metricsAll = {}
    metricsAll['mota'] = np.zeros([1,nJoints+1])
    metricsAll['motp'] = np.zeros([1,nJoints+1])
    metricsAll['pre']  = np.zeros([1,nJoints+1])
    metricsAll['rec']  = np.zeros([1,nJoints+1])

    imgidxfirst = 0
    # iterate over tracking sequences
    # seqidxsUniq = seqidxsUniq[:20]
    accAll = {}
    for i in range(nJoints):
        accAll[i] = mm.MOTAccumulator(auto_id=True)
    nSeq = len(seqidxsUniq)
    for si in range(nSeq):
        print "seqidx: %d/%d" % (si+1,nSeq)
        # extract frames IDs for the sequence
        imgidxs = np.argwhere(seqidxs == seqidxsUniq[si])
        # DEBUG: remove the last frame of each sequence from evaluation due to buggy annotations
        print "DEBUG: remove last frame from eval until annotations are fixed"
        imgidxs = imgidxs[:-1].copy()
        # create an accumulator that will be updated during each frame
        # iterate over frames
        for j in range(len(imgidxs)):
            imgidx = imgidxs[j,0]
            # iterate over joints
            for i in range(nJoints):
                # GT tracking ID
                trackidxGT = motAll[imgidx][i]["trackidxGT"]
                # prediction tracking ID
                trackidxPr = motAll[imgidx][i]["trackidxPr"]
                # distance GT <-> pred part to compute MOT metrics
                # 'NaN' means force no match
                dist = motAll[imgidx][i]["dist"]
                # Call update once per frame
                accAll[i].update(
                    trackidxGT,                 # Ground truth objects in this frame
                    trackidxPr,                 # Detector hypotheses in this frame
                    dist                        # Distances from objects to hypotheses
                )

    # compute metrics per joint for all sequences
    for i in range(nJoints):
        summary = mh.compute(accAll[i], metrics=['num_frames', 'mota', 'motp', 'precision', 'recall'], return_dataframe=False, name='acc')
        metricsAll['mota'][0,i] += summary['mota']*100
        metricsAll['motp'][0,i] += (1-summary['motp'])*100
        metricsAll['pre'][0,i]  += summary['precision']*100
        metricsAll['rec'][0,i]  += summary['recall']*100

    # average metrics over all joints over all sequences
    metricsAll['mota'][0,nJoints] = metricsAll['mota'][0,:nJoints].mean()
    metricsAll['motp'][0,nJoints] = metricsAll['motp'][0,:nJoints].mean()
    metricsAll['pre'][0,nJoints]  = metricsAll['pre'] [0,:nJoints].mean()
    metricsAll['rec'][0,nJoints]  = metricsAll['rec'] [0,:nJoints].mean()

    return metricsAll


def evaluateTracking(gtFramesAll, prFramesAll):

    distThresh = 0.5
    # assign predicted poses to GT poses
    _, _, _, motAll = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute MOT metrics per part
    metricsAll = computeMetrics(gtFramesAll, motAll)

    return metricsAll
