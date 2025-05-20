import cv2
import supervision as sv
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from matplotlib.pyplot import figure
pd.options.mode.chained_assignment = None


def video_resizer(file_in):
    intput_file = "./static/uploads/user_upload.mp4"
    output_file = "./static/uploads/processed/traffic_test_vid.mp4"
    scale_factor = 0.4
    output_fps = 15
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Cannot open the video file.")
        exit()

    # Get original properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # New resolution
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'avc1' or 'X264' if needed
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (new_width, new_height))

    # Frame skipping factor for FPS reduction
    frame_skip = int(original_fps / output_fps)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only every nth frame to reduce FPS
        if frame_count % frame_skip == 0:
            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Write the frame
            out.write(resized_frame)

        frame_count += 1

    # Release everything
    cap.release()
    out.release()

def video_object_identification():
    model = YOLO("./yolov8n.pt")
    csv_sink = sv.CSVSink("./static/results.csv")
    frames_generator = sv.get_video_frames_generator("./static/uploads/processed/traffic_test_vid.mp4")

    with csv_sink as sink:
        for _, frame in enumerate(frames_generator):

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            sink.append(detections, custom_data={'timestamp':_})


def counting_script():
    vehicles = ["car","motorcycle","bus","truck"]

    df = pd.read_csv("results.csv")

    df = df[df["confidence"]>0.6]

    cars = df[df["class_name"].isin(vehicles)]

    # remove stationary objects
    x_mins = cars["x_min"]
    cars["tracker"] = [int(np.round(x/5)) for x in cars["x_min"]]
    counter = Counter(cars["tracker"])
    to_del = [int(x) for x in counter.keys() if counter[x] >40]
    moving_cars = cars[cars["tracker"].isin(to_del) == False]

    # count number of moving objects
    tstamps = list(cars[["timestamp"]].drop_duplicates().loc[:,"timestamp"])

    maxos = []
    times = []

    for i in range(len(tstamps)-1):
        t1 = tstamps[i]
        t2 = tstamps[i+1]
        coords1 = [x for x in moving_cars[moving_cars["timestamp"]==t1].loc[:,"x_min"]]
        coords2 = [x for x in moving_cars[moving_cars["timestamp"]==t2].loc[:,"x_min"]]
        temp = []
        for c1 in coords1:
            for c2 in coords2:
                temp.append(abs(c1-c2))
        if len(temp)>0:
            maxo = max(temp)
        else:
            maxo=0
        maxos.append(maxo)
        times.append(t1)

    w = 20

    data = pd.DataFrame({"time":times,"maxos":maxos})
    data["smooth"] = data[["maxos"]].rolling(w).mean()
    smooth = list(data["smooth"])
    data = data.iloc[w:,:] # get rid of nans made by lag
    smooth = smooth[w:]

    n_detect = 1
    cutoff = 1
    poss_numbers = []

    while n_detect != 0:
        data["detected"] = [x>cutoff for x in smooth]
        data["lagged_detected"] = data[["detected"]].shift(1)
        data = data.fillna(0)
        data["new_one"] = data["detected"] > data["lagged_detected"]
        #print("Number of cars detected:",sum(data["new_one"]))
        n_detect = sum(data["new_one"])
        poss_numbers.append(n_detect)
        cutoff +=1
    counter = Counter(poss_numbers)
    likely_n = max([x for x in counter.keys() if counter[x]>len(poss_numbers)/10])
    print("Number of cars detected is", likely_n)

    figure(figsize=(20, 6), dpi=60)
    plt.plot(data["time"],data["smooth"])
    plt.axhline(y=cutoff, color='b', linestyle='-')
    plt.savefig("plot1.jpg")

    figure(figsize=(8, 6), dpi=60)
    plt.plot(poss_numbers)
    plt.savefig("plot2.jpg")
    return likely_n
