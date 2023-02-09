# test converted tflite model
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import matplotlib
import time
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind':lambda x: "%.4f" % x})


def load_image_into_numpy_array(path, dim=None):
    image = Image.open(path)
    if dim is not None:
        image = image.resize(dim)
    return np.array(image)

def preprocess_image(interpreter, image_path):
    input_details = interpreter.get_input_details()[0]
    _, h, w, c    = input_details["shape"]
    return load_image_into_numpy_array(image_path, (h, w))

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']

    # _, h, w, c = input_details["shape"]
    # interpreter.resize_tensor_input(0, [1, h, w, c])

    image = image / 255.0
    input_scale, input_zero_point = input_details["quantization"]
    if input_scale != 0:
        image = image / input_scale + input_zero_point
    image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(tensor_index, image)


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    # print(f"index={index}, tensor_index={output_details['index']}")
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    # print(f'output={tensor}')
    output_scale, output_zero_point = output_details["quantization"]
    if output_scale != 0:
        tensor = (tensor - output_zero_point) *  output_scale

    return tensor

def run_tf_model(model, label_map, image_path, threshold=0.4):
    IMAGE_SIZE = (5, 5) # Output display size as you want
    detect_fn=tf.saved_model.load(model)

    #Loading the label_map
    category_index=label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)

    image_np = load_image_into_numpy_array(image_path)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    # print(f"{detections['detection_boxes']=}")
    # print(f"{detections['detection_classes']=}")
    # print(f"{detections['detection_scores']=}")
    # print(f"{detections['num_detections']=}")

    image_out = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_out,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=threshold,
        agnostic_mode=False)

    plt.figure(figsize=IMAGE_SIZE, dpi=200)
    plt.axis("off")
    plt.imshow(image_out)
    plt.show()

def run_tflite_model(model, label_map, image_path, output_desc, threshold=0.2):
    interpreter=tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    print(interpreter.get_output_details())
    category_index=label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)
    image_in = preprocess_image(interpreter, image_path)

    start = time.time()
    set_input_tensor(interpreter, image_in)
    interpreter.invoke()
    elapsed = time.time() - start
    print(f'elapsed = {elapsed}')

    scores  = get_output_tensor(interpreter, output_desc['scores']) if 'scores' in output_desc.keys() else None
    boxes   = get_output_tensor(interpreter, output_desc['boxes']) if 'boxes' in output_desc.keys() else None
    count   = int(get_output_tensor(interpreter, output_desc['count'])) if 'count' in output_desc.keys() else None
    classes = np.int32(get_output_tensor(interpreter, output_desc['classes'])) if 'classes' in output_desc.keys() else None
    if 'classes_offset' in output_desc.keys():
        classes = classes + output_desc['classes_offset']

    # print(f'{boxes=}')
    # print(f'{classes=}')
    # print(f'{scores=}')
    # print(f'{count=}')

    #image_out = load_image_into_numpy_array(image_path, (h, w))
    image_out = load_image_into_numpy_array(image_path)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_out,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=threshold,
        agnostic_mode=False)

    plt.figure(figsize=(5, 5), dpi=200)
    plt.axis("off")
    plt.imshow(image_out)
    plt.show()

# run_tflite_model('./.train_outputs/road_signs_2_metadata.tflite',
#                  './.train_outputs/road_signs_label_map.pbtxt',
#                  './.train_outputs/road_test1.jpg',
#                 { 'scores':0, 'boxes':1,'count':2,'classes':3, 'classes_offset':1 },
#                  0.2)

# run_tflite_model('./.train_outputs/road_signs_q8_metadata.tflite',
#                  './.train_outputs/road_signs_label_map.pbtxt',
#                  './.train_outputs/road_test1.jpg',
#                  { 'scores':0, 'boxes':1,'count':2,'classes':3, 'classes_offset':1 },
#                  0.2)

# run_tflite_model('./.train_outputs/ssd_q8_metadata.tflite',
#                  './.train_outputs/mscoco_label_map.pbtxt',
#                  './.train_outputs/ssd_test.jpg',
#                  { 'scores':0, 'boxes':1,'count':2,'classes':3, 'classes_offset':1 },
#                  0.2)

run_tflite_model('./.train_outputs/ssd_fpnlite_q8_metadata.tflite',
                 './.train_outputs/mscoco_label_map.pbtxt',
                 './.train_outputs/ssd_test.jpg',
                 { 'scores':0, 'boxes':1,'count':2,'classes':3, 'classes_offset':1 },
                 0.3)

# run_tf_model(f'{top_dir}/results/saved_model',
#              f'{top_dir}/DeepPiCar/models/object_detection/data/annotations/label_map.pbtxt',
#              f'{top_dir}/DeepPiCar/models/object_detection/data/images/test/2019-04-16-095558.jpg')