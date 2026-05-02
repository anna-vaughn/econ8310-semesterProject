import cv2
import torch
import xml.etree.ElementTree as ET
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_trained_model(weights_path, num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval() 
    return model

#Helper functions
def overlap_percent(ball_box, zone_box):
    x_left = max(ball_box["xtl"], zone_box["xtl"])
    y_top = max(ball_box["ytl"], zone_box["ytl"])
    x_right = min(ball_box["xbr"], zone_box["xbr"])
    y_bottom = min(ball_box["ybr"], zone_box["ybr"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    overlap_area = (x_right - x_left) * (y_bottom - y_top)
    ball_area = (ball_box["xbr"] - ball_box["xtl"]) * (ball_box["ybr"] - ball_box["ytl"])

    if ball_area == 0:
        return 0.0
    return overlap_area / ball_area

def parse_strike_zone(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for track in root.findall(".//track"):
        label = track.attrib.get("label", "").lower()
        if label in ["strikezone", "strike_zone", "strike zone"]:
            box = track.find("box")
            if box is not None:
                return {
                    "xtl": float(box.attrib["xtl"]),
                    "ytl": float(box.attrib["ytl"]),
                    "xbr": float(box.attrib["xbr"]),
                    "ybr": float(box.attrib["ybr"]),
                }
    return None

def calculate_centroid(box):
    """Calculates the center (x, y) of a bounding box [xmin, ymin, xmax, ymax]."""
    xmin, ymin, xmax, ymax = box
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    return (cx, cy)

def evaluate_video(model, video_path, strike_zone_xml, output_path, confidence_threshold=0.5):
    device = torch.device('cpu')
    model.to(device)

    #Load the strike zone from the xml parser
    strike_zone = parse_strike_zone(strike_zone_xml)
    if strike_zone is None:
        print("Error: Could not find strike zone in XML.")
        return
        
    sz_xmin, sz_ymin = int(strike_zone["xtl"]), int(strike_zone["ytl"])
    sz_xmax, sz_ymax = int(strike_zone["xbr"]), int(strike_zone["ybr"])

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    trajectory = [] #Holds centroids to draw the line
    pitch_result = "BALL" #Default state

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(rgb_frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            predictions = model([img_tensor])[0]

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()

        best_score = 0
        best_box = None

        #Find the highest confidence ball in the frame
        for box, score in zip(boxes, scores):
            if score > confidence_threshold and score > best_score:
                best_score = score
                best_box = box

        if best_box is not None:
            xmin, ymin, xmax, ymax = map(int, best_box)
            cx, cy = calculate_centroid(best_box)
            trajectory.append((cx, cy)) #Save for line drawing
            
            #Format the model's box 
            ball_box_dict = {"xtl": xmin, "ytl": ymin, "xbr": xmax, "ybr": ymax}
            
            #Check if this frame is a strike
            overlap = overlap_percent(ball_box_dict, strike_zone)
            if overlap >= 0.33:
                pitch_result = "STRIKE"
                #Highlight the ball box in yellow if it's currently inside the strike zone
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
            else:
                #Normal red box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        #Draw the visual Trajectory Line
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i-1], trajectory[i], (0, 255, 0), 3)
            
        #Draw the Strike Zone
        cv2.rectangle(frame, (sz_xmin, sz_ymin), (sz_xmax, sz_ymax), (255, 0, 0), 2)
        
        #Display the result
        cv2.putText(frame, f"Result: {pitch_result}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Final Pitch Result: {pitch_result}")
    print(f"Annotated video saved here: {output_path}")

if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    
    model_weights = PROJECT_ROOT / 'baseball_weights_throwaway.pth' 
    trained_model = load_trained_model(str(model_weights))

    #UPDATE THIS PATH TO WHEREVER YOU HAVE VIDEOS DOWNLOADED
    input_video = r''

    if input_video == '':
        print("Hey! Please add your local video path to the 'input_video' variable.")
    
    else:

    #Find the SZone file in the "30 to 39" folder 
        annotation_folder = PROJECT_ROOT / "Baseball Annotations" / "30 to 39"
        szone_files = list(annotation_folder.glob("*SZone.xml"))
        
        if not szone_files:
            print(f"Error: Could not find an SZone.xml file in {annotation_folder}")
        else:
            strike_zone_xml = str(szone_files[0]) # Grabs the correct SZone file 
            
            #Setup output path
            output_dir = PROJECT_ROOT / "Output"
            output_dir.mkdir(exist_ok=True)
            output_video = str(output_dir / 'annotated_IMG_0031.mp4')
            
            #Evaluate
            print(f"Evaluating {input_video}...")
            print(f"Using Strike Zone from: {strike_zone_xml}")
            evaluate_video(trained_model, input_video, strike_zone_xml, output_video)

#How to run:
#evaluate_video(model, 'video.mov', 'video_SZone.xml', 'output.mp4')
