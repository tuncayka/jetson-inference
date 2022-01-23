import time
import datetime
import argparse
import sys
import uuid
import jetson.inference
import jetson.utils
try:
    import requests
except:
    import os
    os.system("pip3 install requests")
finally:
    import requests

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage() +jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--api", type=str, default=None, help="api address for sending detections.")


try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# api_url = "http://192.168.1.105:8080/test"
api_url = opt.api
overlay="box,labels,conf"
net = jetson.inference.detectNet("ped-100", threshold=0.5)
#'--model=models/fruit/ssd-mobilenet.onnx', '--labels=models/fruit/labels.txt'
#, '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'
argv = ['--model=tlt_dashcamnet_pruned_v1.0/resnet18_dashcamnet_pruned.etlt'
        , '--labels=tlt_dashcamnet_pruned_v1.0/labels.txt'
        , '--output-cvg=boxes'
        , '--output-bbox=boxes'
        , '--input-blob=input_0'
]

# create video sources
camera = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)

# net = jetson.inference.detectNet(argv=argv, threshold=0.5)
# net = jetson.inference.detectNet("facenet", threshold=0.5)
# camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
# camera = jetson.utils.videoSource("/dev/video1")
# camera = jetson.utils.videoSource("rtsp://10.0.202.59/live/ch00_0")
#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

# while display.IsStreaming():
while True:

    time.sleep(1)
    img = camera.Capture()
    detections = net.Detect(img, overlay=overlay)

    dt = "{}".format(datetime.datetime.now())

    json = dict()
    json['timestamp'] = dt
    json['machine_id'] = ':'.join(['{:02x}'.format((uuid.getnode() >> ele) & 0xff) for ele in range(0,8*6,8)][::-1])
    json['num_of_people'] = len(detections)
    json_detections = []

    # print the detections
    print("detected {:d} objects in image".format(len(detections)))

    for detection in detections:
        d = dict()
        d['class_id'] = detection.ClassID
        d['class_text'] = net.GetClassDesc(detection.ClassID)
        d['confidence'] = detection.Confidence
        d['left'] = detection.Left
        d['top'] = detection.Top
        d['right'] = detection.Right
        d['bottom'] = detection.Bottom
        d['width'] = detection.Width
        d['height'] = detection.Height
        d['area'] = detection.Area
        d['center'] = detection.Center
        json_detections.append(d)
        print(detection)

    json['detections'] = json_detections
    try:
        r = requests.post(api_url, json=json)
    except:
        print("API Error. ")
        pass

    #display.Render(img)
    #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))