import os
import urllib
import tarfile
import tensorflow as tf
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util
import cv2
import numpy as np
import imageio

MODEL_NAME = "SIBI-classifier"

#Path ke file model yang sudah dibuat.
PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, 'grayscale800.30000_90.10_0.0005_batch.16_frozen_inference_graph.pb')
#Daftar string yang digunakan untuk menambahkan label yang benar untuk setiap kotak.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'grayscale800.30000_90.10_0.0005_batch.16_sign-language_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Using URLLIB to initialize from smartphone camera 
url='http://192.168.137.145:8080/shot.jpg'

# Using OpenCV to initialize the webcam
#vs = cv2.VideoCapture(0)

# adjust fps according to your needs if saved video too slow or fast
writer = imageio.get_writer('output_smartphone.mp4', fps=1)
with detection_graph.as_default():
    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        while True:
            imgResp=urllib.request.urlopen(url)
            imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
            img=cv2.imdecode(imgNp,-1)
            image_np=img
            
            #ret, image_np = vs.read()
            
            dimension = (800,600)
            image_np = cv2.resize(image_np, dimension)
            #image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2GRAY)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=10, max_boxes_to_draw=1)
            
            writer.append_data(image_np)
            cv2.imshow('Sign Language', image_np)
            if cv2.waitKey(1) & 0xFF == ord('q'): #13 is the Enter Key
                break
            
# Release camera and close windows
writer.close()
#vs.release()
cv2.destroyAllWindows()