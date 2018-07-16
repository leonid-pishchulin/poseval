#!/usr/bin/env python
"""Convert between COCO and PoseTrack2017 format."""
# pylint: disable=too-many-branches, too-many-locals, bad-continuation
import json
import logging
import os
import os.path as path

import click
import numpy as np
import tqdm

from posetrack18_id2fname import posetrack18_fname2id, posetrack18_id2fname

LOGGER = logging.getLogger(__name__)
POSETRACK18_LM_NAMES = [
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "neck",
    "nose",
    "head_top",
]
COCO_TO_MPII = [None, None, None, None, None, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]

EXPORT_IMAGE_COUNTER = 0
EXPORT_PERSON_COUNTER = 0
SCORE_WARNING_EMITTED = False


def json_default(val):
    """Serialization workaround
    https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python."""
    if isinstance(val, np.int64):
        return int(val)
    raise TypeError


class Video:

    """
    A PoseTrack sequence.

    Parameters
    ==========

    video_id: str.
      A five or six digit number, potentially with leading zeros, identifying the
      PoseTrack video.
    """

    def __init__(self, video_id):
        self.posetrack_video_id = video_id  # str.
        self.frames = []  # list of Image objects.

    def to_new(self):
        """Return a dictionary representation for the PoseTrack18 format."""
        global EXPORT_IMAGE_COUNTER  # pylint: disable=global-statement
        result = {"images": [], "annotations": []}
        for image in self.frames:
            image_json = image.to_new()
            image_json["vid_id"] = self.posetrack_video_id
            image_json["nframes"] = len(self.frames)
            image_json["id"] = EXPORT_IMAGE_COUNTER
            result["images"].append(image_json)
            for person in image.people:
                person_json = person.to_new()
                person_json["image_id"] = EXPORT_IMAGE_COUNTER
                result["annotations"].append(person_json)
            EXPORT_IMAGE_COUNTER += 1
        # Write the 'categories' field.
        result["categories"] = [
            {
                "supercategory": "person",
                "name": "person",
                "skeleton": [
                    [14, 13],
                    [13, 12],
                    [12, 9],
                    [9, 10],
                    [10, 11],
                    [9, 3],
                    [3, 4],
                    [4, 5],
                    [12, 8],
                    [8, 7],
                    [7, 6],
                    [8, 2],
                    [2, 1],
                    [1, 0],
                ],
                "keypoints": POSETRACK18_LM_NAMES,
                "id": 1,
            }
        ]
        return result

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        res = {"annolist": []}
        for image in self.frames:
            for person in image.people:
                elem = {}
                elem["image"] = [{"name": image.posetrack_filename}]
                elem["annorect"] = [person.to_old()]
                res["annolist"].append(elem)
        return res

    @classmethod
    def from_old(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        assert "annolist" in track_data.keys(), "Wrong format!"
        video = None
        for image_info in track_data["annolist"]:
            image = Image.from_old(image_info)
            if not video:
                video = Video(
                    path.basename(path.dirname(image.posetrack_filename)).split("_")[0]
                )
            else:
                assert (
                    video.posetrack_video_id
                    == path.basename(path.dirname(image.posetrack_filename)).split("_")[
                        0
                    ]
                )
            video.frames.append(image)
        return [video]

    @classmethod
    def from_new(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        image_id_to_can_info = {}
        video_id_to_video = {}
        assert len(track_data["categories"]) == 1
        assert track_data["categories"][0]["name"] == "person"
        assert len(track_data["categories"][0]["keypoints"]) in [15, 17]
        if len(track_data["categories"][0]["keypoints"]) == 17:
            conversion_table = COCO_TO_MPII
        else:
            conversion_table = []
            for lm_name in track_data["categories"][0]["keypoints"]:
                if lm_name not in POSETRACK18_LM_NAMES:
                    conversion_table.append(None)
                else:
                    conversion_table.append(POSETRACK18_LM_NAMES.index(lm_name))
            for lm_idx, lm_name in enumerate(POSETRACK18_LM_NAMES):
                assert lm_idx in conversion_table, "Landmark `%s` not found." % (
                    lm_name
                )
        videos = []
        for person_info in track_data["annotations"]:
            image_id = person_info["image_id"]
            if image_id in image_id_to_can_info:
                image = image_id_to_can_info[image_id]
            else:
                image = Image.from_new(track_data, image_id)
                video_id = path.basename(path.dirname(image.posetrack_filename)).split(
                    "_"
                )[0]
                #  video_id = video_id_str[1:-3]
                if video_id in video_id_to_video.keys():
                    video = video_id_to_video[video_id]
                else:
                    video = Video(video_id)
                    video_id_to_video[video_id] = video
                    videos.append(video)
                video.frames.append(image)
                image_id_to_can_info[image_id] = image
            image.people.append(Person.from_new(person_info, conversion_table))
        return videos


class Person:

    """
    A PoseTrack annotated person.

    Parameters
    ==========

    track_id: int
      Unique integer representing a person track.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.landmarks = None  # None or list of dicts with 'score', 'x', 'y', 'id'.
        self.rect = None  # None or dict with 'x1', 'x2', 'y1' and 'y2'.
        self.score = None  # None or float.

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The field 'image_id' must be added to the result.
        """
        global EXPORT_PERSON_COUNTER  # pylint: disable=global-statement
        EXPORT_PERSON_COUNTER += 1
        keypoints = []
        scores = []
        write_scores = (
            len([1 for lm_info in self.landmarks if "score" in lm_info.keys()]) > 0
        )
        for landmark_id in range(15):
            try:
                landmark_info = [
                    lm for lm in self.landmarks if lm["id"] == landmark_id
                ][0]
            except IndexError:
                landmark_info = {"x": 0, "y": 0}
            is_visible = 1
            if "is_visible" in landmark_info.keys():
                is_visible = landmark_info["is_visible"]
            keypoints.extend([landmark_info["x"], landmark_info["y"], is_visible])
            if "score" in landmark_info.keys():
                scores.append(landmark_info["score"])
            elif write_scores:
                LOGGER.warning("Landmark with missing score info detected. Using 0.")
                scores.append(0.)
        return {
            "track_id": self.track_id,
            "bbox": [
                self.rect["x1"],
                self.rect["y1"],
                self.rect["x2"] - self.rect["x1"],
                self.rect["y2"] - self.rect["y1"],
            ],
            "category_id": 1,
            "id": EXPORT_PERSON_COUNTER,
            "keypoints": keypoints,
            "scores": scores,
            # image_id added later.
        }

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        keypoints = []
        for landmark_info in self.landmarks:
            point = {
                "id": [landmark_info["id"]],
                "x": [landmark_info["x"]],
                "y": [landmark_info["y"]],
            }
            if "score" in landmark_info.keys():
                point["score"] = [landmark_info["score"]]
            if "is_visible" in landmark_info.keys():
                point["is_visible"] = [landmark_info["is_visible"]]
            keypoints.append({"point": [point]})
        ret = {
            "x1": [self.rect["x1"]],
            "x2": [self.rect["x2"]],
            "y1": [self.rect["y1"]],
            "y2": [self.rect["y2"]],
            "track_id": [self.track_id],
            "annopoints": keypoints,
        }
        if self.score:
            ret["score"] = [self.score]
        return ret

    @classmethod
    def from_old(cls, person_info):
        """Parse a dictionary representation from the PoseTrack17 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"][0])
        assert len(person_info["track_id"]) == 1, "Invalid format!"
        rect = {}
        rect["x1"] = person_info["x1"][0]
        assert len(person_info["x1"]) == 1, "Invalid format!"
        rect["x2"] = person_info["x2"][0]
        assert len(person_info["x2"]) == 1, "Invalid format!"
        rect["y1"] = person_info["y1"][0]
        assert len(person_info["y1"]) == 1, "Invalid format!"
        rect["y2"] = person_info["y2"][0]
        assert len(person_info["y2"]) == 1, "Invalid format!"
        person.rect = rect
        try:
            person.score = person_info["score"][0]
            assert len(person_info["score"]) == 1, "Invalid format!"
        except KeyError:
            pass
        person.landmarks = []
        if not person_info["annopoints"]:
            return person
        for landmark_info in person_info["annopoints"][0]["point"]:
            lm_dict = {
                "y": landmark_info["y"][0],
                "x": landmark_info["x"][0],
                "id": landmark_info["id"][0],
            }
            if "score" in landmark_info.keys():
                lm_dict["score"] = landmark_info["score"][0]
                assert len(landmark_info["score"]) == 1, "Invalid format!"
            elif not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
            if "is_visible" in landmark_info.keys():
                lm_dict["is_visible"] = landmark_info["is_visible"][0]
            person.landmarks.append(lm_dict)
            assert (
                len(landmark_info["x"]) == 1
                and len(landmark_info["y"]) == 1
                and len(landmark_info["id"]) == 1
            ), "Invalid format!"
        return person

    @classmethod
    def from_new(cls, person_info, conversion_table):
        """Parse a dictionary representation from the PoseTrack18 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"])
        rect = {}
        rect["x1"] = person_info["bbox"][0]
        rect["x2"] = person_info["bbox"][0] + person_info["bbox"][2]
        rect["y1"] = person_info["bbox"][1]
        rect["y2"] = person_info["bbox"][1] + person_info["bbox"][3]
        person.rect = rect
        try:
            person.score = person_info["score"]
        except KeyError:
            if not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
        person.landmarks = []
        for landmark_idx, landmark_info in enumerate(
            np.array(person_info["keypoints"]).reshape(len(conversion_table), 3)
        ):
            landmark_idx_can = conversion_table[landmark_idx]
            if landmark_idx_can:
                person.landmarks.append(
                    {
                        "y": landmark_info[1],
                        "x": landmark_info[0],
                        "id": landmark_idx_can,
                        "is_visible": landmark_info[2],
                    }
                )
        return person


class Image:

    """An image with annotated people on it."""

    def __init__(self, filename, frame_id):
        self.posetrack_filename = filename
        self.frame_id = frame_id
        self.people = []

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The field 'vid_id' must still be added.
        """
        return {
            "file_name": self.posetrack_filename,
            "has_no_densepose": True,
            "is_labeled": False,
            "frame_id": self.frame_id,
            # vid_id and nframes are inserted later.
        }

    @classmethod
    def from_old(cls, json_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        posetrack_filename = json_data["image"][0]["name"]
        assert len(json_data["image"]) == 1, "Invalid format!"
        old_seq_fp = path.basename(path.dirname(posetrack_filename))
        old_frame_id = int(path.basename(posetrack_filename).split(".")[0]) - 1
        frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        image = Image(posetrack_filename, frame_id)
        for person_info in json_data["annorect"]:
            image.people.append(Person.from_old(person_info))
        return image

    @classmethod
    def from_new(cls, track_data, image_id):
        """Parse a dictionary representation from the PoseTrack18 format."""
        image_info = [
            image_info
            for image_info in track_data["images"]
            if image_info["id"] == image_id
        ][0]
        posetrack_filename = image_info["file_name"]
        # license, coco_url, height, width, date_capture, flickr_url, id are lost.
        old_seq_fp = path.basename(path.dirname(posetrack_filename))
        old_frame_id = int(path.basename(posetrack_filename).split(".")[0]) - 1
        frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        return Image(posetrack_filename, frame_id)


@click.command()
@click.argument(
    "in_fp", type=click.Path(exists=True, readable=True, dir_okay=True, file_okay=True)
)
@click.option(
    "--out_fp",
    type=click.Path(exists=False, writable=True, file_okay=False),
    default="converted",
    help="Write the results to this folder (may not exist). Default: converted.",
)
def cli(in_fp, out_fp="converted"):
    """Convert between PoseTrack18 and PoseTrack17 format."""
    LOGGER.info("Converting `%s` to `%s`...", in_fp, out_fp)
    if in_fp.endswith(".zip") and path.isfile(in_fp):
        LOGGER.info("Unzipping...")
        import zipfile
        import tempfile

        unzip_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(in_fp, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        in_fp = unzip_dir
        LOGGER.info("Done.")
    else:
        unzip_dir = None
    if path.isfile(in_fp):
        track_fps = [in_fp]
    else:
        track_fps = [
            path.join(in_fp, track_fp)
            for track_fp in os.listdir(in_fp)
            if track_fp.endswith(".json")
        ]
    LOGGER.info("Identified %d track files.", len(track_fps))
    assert path.isfile(track_fps[0]), "`%s` is not a file!" % (track_fps[0])
    with open(track_fps[0], "r") as inf:
        first_track = json.load(inf)
    # Determine format.
    old_to_new = False
    if "annolist" in first_track.keys():
        old_to_new = True
        LOGGER.info("Detected PoseTrack17 format. Converting to 2018...")
    else:
        assert "images" in first_track.keys(), "Unknown image format. :("
        LOGGER.info("Detected PoseTrack18 format. Converting to 2017...")

    videos = []
    LOGGER.info("Parsing data...")
    for track_fp in tqdm.tqdm(track_fps):
        with open(track_fp, "r") as inf:
            track_data = json.load(inf)
        if old_to_new:
            videos.extend(Video.from_old(track_data))
        else:
            videos.extend(Video.from_new(track_data))
    LOGGER.info("Writing data...")
    if not path.exists(out_fp):
        os.mkdir(out_fp)
    for video in tqdm.tqdm(videos):
        target_fp = path.join(
            out_fp,
            video.posetrack_video_id + ".json"
            # posetrack18_id2fname(video.posetrack_video_id)[0] + ".json"  # TODO: reenable conversion.
        )
        if old_to_new:
            converted_json = video.to_new()
        else:
            converted_json = video.to_old()
        with open(target_fp, "w") as outf:
            json.dump(converted_json, outf, default=json_default)
    if unzip_dir:
        LOGGER.debug("Deleting temporary directory...")
        os.unlink(unzip_dir)
    LOGGER.info("Done.")


logging.basicConfig(level=logging.DEBUG)
cli()  # pylint: disable=no-value-for-parameter
