"""
count_islands.py -- count the islands of every class in a segmentation file.


WHAT IS AN ISLAND?
------------------
A segmentation file stores one whole number per voxel: 0 for background,
1, 2, 3, ... for the organs. Take all the voxels of one class and ask which of
them touch each other. Each group of voxels that hang together is an island
(the usual name is a "connected component").

    class 1 in a slice, "." = background, "#" = class 1

        . . . . . . . . . .
        . # # . . . # # # .      two islands: the left blob
        . # # . . . # # # .      and the right blob do not touch
        . . . . . . # # # .


WHY COUNT THEM?
---------------
  * Anatomy check. A trachea is one tube, so its class should be one island.
    Several islands mean either a labelling mistake or an organ cut in two.
  * Noise check. A model prediction often has a correct organ plus a handful
    of stray voxels elsewhere. Those stray voxels are tiny extra islands, and
    counting them tells you how noisy the prediction is.
  * In this dataset. GT class 1 holds the esophagus and the aorta merged
    together (see task01). If they did not touch, class 1 would be two islands
    and separating them would be free. So the island count answers directly:
    can I get the two organs apart by connectivity alone?


WHAT DOES "TOUCH" MEAN? THE CONNECTIVITY CHOICE
-----------------------------------------------
This is the part that trips people up. In 3D, a voxel has three kinds of
neighbours, and you have to choose which ones count as touching:

    6 neighbours  (face)    share a whole face with the voxel        (up, down, left, right, front, back)
    18 neighbours (edge)    the 6 above, plus the ones sharing an edge
    26 neighbours (corner)  every voxel in the surrounding 3x3x3 cube, corners included

In 2D the same choice is called 4-connectivity and 8-connectivity:

        . # .           the two "#" below share only a corner:
        # . .           8-connected (one island), but not 4-connected (two islands)

        # .
        . #

The looser the rule, the more things count as connected, so the island count
can only go down as you go from 6 to 26. That is why this script always prints
both: if the counts disagree, the parts touch only diagonally, which is a weak
connection worth knowing about. In this dataset that is exactly what happens in
patients 02, 09 and 16.

One caveat for CT: voxels here are about 1 x 1 x 2.5 mm, so they are not cubes.
A diagonal step in the slice plane is about 1.4 mm, while a step to the next
slice is 2.5 mm. "Touching" is therefore not the same physical distance in
every direction.


WHAT DOES "LARGE" MEAN?
-----------------------
A single stray voxel is an island too, so a raw island count is easily
dominated by noise. This script also reports how many islands are large, which
here means holding at least 1% of that class's voxels. So "2 large islands"
means two real structures, not one organ plus specks.


HOW TO RUN
----------
    PY=/home/tai/work-my-projects/workspace-master.course.y2.block01-01-AI4MI/workspace-group.project-01-mid/ai4mi_project/.venv/bin/python

    $PY count_islands.py                              # the two Patient_07 files
    $PY count_islands.py patient-02-GT.nii.gz         # any file(s) you name
    $PY count_islands.py *.nii.gz --json islands.json # also save the numbers
"""

import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Settings. Paths are hardcoded on purpose: this script is for one dataset.
# ---------------------------------------------------------------------------
DEFAULT_FILES = [
    "/home/tai/work-my-projects/workspace-master.course.y2.block01-01-AI4MI"
    "/workspace-group.project-01-mid/ai4mi_project/data/segthor_part1/train/Patient_07/GT.nii.gz",
    "/home/tai/work-my-projects/workspace-master.course.y2.block01-01-AI4MI"
    "/workspace-group.project-01-mid/ai4mi_project/data/segthor_part1/train/Patient_07/GT2.nii.gz",
]

# An island counts as large when it holds at least this share of its class.
MIN_LARGE_FRACTION = 0.01

# The two neighbourhoods explained at the top of this file. scipy wants them as
# a 3x3x3 array of True/False saying which of the 26 surrounding voxels count.
FACE_NEIGHBOURS = ndimage.generate_binary_structure(3, 1)  # 6 of them
CORNER_NEIGHBOURS = np.ones((3, 3, 3), bool)  # all 26

# Class meaning in SegTHOR. GT has classes 1-3 only, with the aorta merged into
# class 1; GT2 keeps the aorta as its own class 4.
CLASS_NAMES = {
    1: "esophagus",
    2: "heart",
    3: "trachea",
    4: "aorta",
}


def island_sizes(class_mask, neighbourhood):
    """Sizes of the islands in `class_mask`, biggest first.

    ndimage.label does the real work: it hands back a volume where the voxels
    of the first island are all 1, the second island all 2, and so on. Counting
    how often each number appears therefore gives the island sizes.
    """
    numbered_islands, how_many = ndimage.label(class_mask, structure=neighbourhood)
    if how_many == 0:
        return np.array([], dtype=int)
    voxels_per_island = np.bincount(numbered_islands.ravel())[1:]  # drop the background count
    return np.sort(voxels_per_island)[::-1]


# ---------------------------------------------------------------------------
# Report one file: for each class, count islands under both neighbourhoods.
# ---------------------------------------------------------------------------
def report_one_file(path):
    label_volume = np.asarray(nib.load(path).dataobj)
    classes_present = [int(c) for c in np.unique(label_volume) if c != 0]

    # A file without class 4 is a GT file, where the aorta lives inside class 1.
    aorta_is_merged_into_class_1 = 4 not in classes_present

    print(f"== {path}")
    print(f"   classes present: {classes_present} (0 = background, left out)")
    if aorta_is_merged_into_class_1:
        print("   no class 4 here, so this is a GT file: its class 1 is esophagus + aorta merged")

    numbers_for_json = {}
    for class_id in classes_present:
        class_mask = label_volume == class_id
        total_voxels = int(class_mask.sum())
        threshold = MIN_LARGE_FRACTION * total_voxels

        sizes_by_face = island_sizes(class_mask, FACE_NEIGHBOURS)
        sizes_by_corner = island_sizes(class_mask, CORNER_NEIGHBOURS)
        large_by_face = sizes_by_face[sizes_by_face >= threshold]
        large_by_corner = sizes_by_corner[sizes_by_corner >= threshold]

        organ = CLASS_NAMES.get(class_id, "unknown class")
        if class_id == 1 and aorta_is_merged_into_class_1:
            organ = "esophagus + aorta merged"
        print(f"   class {class_id}: {total_voxels:,} voxels -- {organ}")
        print(f"     touching by face   (6-conn) : {len(large_by_face)} large island(s)"
              f" of {len(sizes_by_face)} total, sizes {[int(s) for s in large_by_face]}")
        print(f"     touching by corner (26-conn): {len(large_by_corner)} large island(s)"
              f" of {len(sizes_by_corner)} total, sizes {[int(s) for s in large_by_corner]}")

        # Say out loud what those two lines mean together.
        if len(large_by_face) != len(large_by_corner):
            print(f"     -> {len(large_by_face)} large islands by face, but {len(large_by_corner)} by corner:"
                  " the parts touch only diagonally, so they are barely connected.")
            print("        For class 1 that is good news: counting with face connectivity"
                  " already separates the two merged organs.")
        elif len(large_by_face) == 1:
            print(f"     -> ONE large island, holding {sizes_by_face[0] / total_voxels:.1%}"
                  " of the class, under both neighbourhoods.")
        else:
            print(f"     -> {len(large_by_face)} large islands under both neighbourhoods:"
                  " these really are separate structures.")

        # Stray voxels: islands too small to be a real structure.
        stray_islands = len(sizes_by_face) - len(large_by_face)
        if stray_islands:
            print(f"        plus {stray_islands} island(s) smaller than {MIN_LARGE_FRACTION:.0%}"
                  f" of the class ({int(sizes_by_face[len(large_by_face):].sum()):,} voxels in total),"
                  " which look like labelling noise.")

        numbers_for_json[class_id] = {
            "voxels": total_voxels,
            "face": {
                "islands": int(len(sizes_by_face)),
                "large_islands": int(len(large_by_face)),
                "large_sizes": [int(s) for s in large_by_face],
                "biggest_share": float(sizes_by_face[0] / total_voxels),
            },
            "corner": {
                "islands": int(len(sizes_by_corner)),
                "large_islands": int(len(large_by_corner)),
                "large_sizes": [int(s) for s in large_by_corner],
                "biggest_share": float(sizes_by_corner[0] / total_voxels),
            },
        }

    print()
    return numbers_for_json


# ---------------------------------------------------------------------------
# Command line: the files to look at, and an optional --json to save numbers.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    arguments = sys.argv[1:]

    json_path = None
    if "--json" in arguments:
        json_path = arguments[arguments.index("--json") + 1]
        arguments.remove("--json")
        arguments.remove(json_path)

    files_to_check = arguments or DEFAULT_FILES
    if not arguments:
        print("no files given, so looking at the two Patient_07 files\n")

    all_numbers = {path: report_one_file(path) for path in files_to_check}

    if json_path:
        with open(json_path, "w") as json_file:
            json.dump(all_numbers, json_file, indent=2)
        print("wrote", json_path)
