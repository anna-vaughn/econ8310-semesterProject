from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

### DATA Path Setup

PROJECT_ROOT = Path(__file__).resolve().parent

### Each subfolder should contain 1 strike zone XML (*SZone.xml); 1+ baseball tracking XML files (IMG_*.xml)
BASE_DIR = PROJECT_ROOT / "Baseball Annotations"

### Define where output CSV files will be saved
OUTPUT_DIR = PROJECT_ROOT / "Output"

### Debug prints to verify correct path resolution
print("Script location:", Path(__file__).resolve())
print("Project root:", PROJECT_ROOT)
print("Base dir:", BASE_DIR)
print("Base dir exists:", BASE_DIR.exists())

if not BASE_DIR.exists():
    raise FileNotFoundError(f"Could not find Baseball Annotations folder at: {BASE_DIR}")

### Create output folder if it does not already exist
OUTPUT_DIR.mkdir(exist_ok=True)


### Calculates what percentage of the baseball bounding box overlaps with the strike zone bounding box, 33% of the baseball is inside the strike zone
def overlap_percent(ball_box, zone_box):

    ### Determine the overlapping rectangle boundaries
    x_left = max(ball_box["xtl"], zone_box["xtl"])
    y_top = max(ball_box["ytl"], zone_box["ytl"])
    x_right = min(ball_box["xbr"], zone_box["xbr"])
    y_bottom = min(ball_box["ybr"], zone_box["ybr"])

    ### If no overlap exists, return 0
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    ### Compute overlapping area
    overlap_area = (x_right - x_left) * (y_bottom - y_top)

    ### Compute total baseball bounding box area
    ball_area = (ball_box["xbr"] - ball_box["xtl"]) * (ball_box["ybr"] - ball_box["ytl"])

    if ball_area == 0:
        return 0.0

    return overlap_area / ball_area


### PARSE STRIKE ZONE XML, Assumes one strike zone per folder
def parse_strike_zone(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for track in root.findall(".//track"):

        ### Normalize label 
        label = track.attrib.get("label", "").lower()

        ### Identify strike zone track
        if label in ["strikezone", "strike_zone", "strike zone"]:

            ### Extract bounding box
            box = track.find("box")

            if box is not None:
                return {
                    "xtl": float(box.attrib["xtl"]),
                    "ytl": float(box.attrib["ytl"]),
                    "xbr": float(box.attrib["xbr"]),
                    "ybr": float(box.attrib["ybr"]),
                }

    return None


### PARSE BASEBALL XML, baseball objects, visible frames (outside == 0), moving baseballs (moving == true)
def parse_baseballs(xml_file, strike_zone, folder_name):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    source_video = root.findtext(".//source")

    rows = []

    for track in root.findall(".//track"):
        label = track.attrib.get("label", "").lower()

        ### Skip non-baseball objects
        if label != "baseball":
            continue

        for box in track.findall("box"):

            outside = int(box.attrib.get("outside", 0))
            if outside == 1:
                continue

            moving_attr = box.find("attribute[@name='moving']")
            moving = moving_attr.text.lower() == "true" if moving_attr is not None else False

            ### Skip non-moving baseballs
            if not moving:
                continue

            ### Build baseball bounding box
            ball_box = {
                "xtl": float(box.attrib["xtl"]),
                "ytl": float(box.attrib["ytl"]),
                "xbr": float(box.attrib["xbr"]),
                "ybr": float(box.attrib["ybr"]),
            }

            ### Calculate overlap with strike zone
            overlap = overlap_percent(ball_box, strike_zone)

            ### Classify as strike if ≥ 33% overlap
            is_strike = overlap >= 0.33

            rows.append({
                "folder": folder_name,
                "xml_file": xml_file.name,
                "video": source_video,
                "frame": int(box.attrib["frame"]),
                "ball_xtl": ball_box["xtl"],
                "ball_ytl": ball_box["ytl"],
                "ball_xbr": ball_box["xbr"],
                "ball_ybr": ball_box["ybr"],
                "strikezone_xtl": strike_zone["xtl"],
                "strikezone_ytl": strike_zone["ytl"],
                "strikezone_xbr": strike_zone["xbr"],
                "strikezone_ybr": strike_zone["ybr"],
                "overlap_percent": overlap,
                "is_strike": is_strike,
            })

    return rows


### MAIN Pipeline

all_rows = []

for folder in BASE_DIR.iterdir():

    if not folder.is_dir():
        continue

    szone_files = list(folder.glob("*SZone.xml"))

    if len(szone_files) == 0:
        print(f"Skipping {folder.name}: no SZone file found")
        continue

    strike_zone = parse_strike_zone(szone_files[0])

    if strike_zone is None:
        print(f"Skipping {folder.name}: no Strikezone label found")
        continue

    ### Identify baseball XML files (exclude strike zone files)
    baseball_files = [
        f for f in folder.glob("*.xml")
        if "SZone" not in f.name
    ]

    ### Parse each baseball file
    for baseball_file in baseball_files:
        rows = parse_baseballs(
            xml_file=baseball_file,
            strike_zone=strike_zone,
            folder_name=folder.name
        )

        ### Add results to full dataset
        all_rows.extend(rows)


df = pd.DataFrame(all_rows)


### SAVE Output

detail_path = OUTPUT_DIR / "strike_analysis_detail.csv"
df.to_csv(detail_path, index=False)


### SUMMMARY dataset

summary = (
    df.groupby(["folder", "video"])
    .agg(
        total_moving_ball_frames=("frame", "count"),
        strike_frames=("is_strike", "sum"),
        avg_overlap_percent=("overlap_percent", "mean"),
        max_overlap_percent=("overlap_percent", "max"),
    )
    .reset_index()
)

### Compute strike rate per video
summary["strike_frame_rate"] = (
    summary["strike_frames"] / summary["total_moving_ball_frames"]
)

### Assign final pitch result (any strike frame = strike)
summary["pitch_result"] = summary["strike_frames"].apply(
    lambda x: "Strike" if x > 0 else "Ball"
)

summary_path = OUTPUT_DIR / "strike_analysis_summary.csv"
summary.to_csv(summary_path, index=False)


### PRINT Resutls

print("\nDetailed results:")
print(df.head())

print("\nSummary:")
print(summary.head())

print("\nOverall strike count:")
print(df["is_strike"].value_counts())

print(f"\nSaved detail file to: {detail_path}")
print(f"Saved summary file to: {summary_path}")
