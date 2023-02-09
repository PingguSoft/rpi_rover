import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import load_model
from object_detection.utils import label_map_util
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils


_width = 320
_height = 320
_dataset = None

_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD  = 127.5

#
#
# def aaa():
#     # convert checkpoint file into TFLite compatible graph
#     ssd_use_regular_nms = True
#     centernet_include_keypoints = False
#     keypoint_label_map_path = None
#     max_detections = 20
#
#     pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
#
#     with tf.io.gfile.GFile(pipelineFilePath, 'r') as f:
#         text_format.Parse(f.read(), pipeline_config)
#
#     export_tflite_graph_lib_tf2.export_tflite_model(
#         pipeline_config, checkPointFileDir, outputDir,
#         max_detections, ssd_use_regular_nms,
#         centernet_include_keypoints, keypoint_label_map_path)
#     print("Created tflite compatible graph from checkpoint file")
#     # now build a tflite model file in outputDir
#     # tf.compat.v1.disable_eager_execution()
#     converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(outputDir, 'saved_model'))
#     converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
#         tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
#     ]
#     tflite_model = converter.convert()
#
#     self.TFLITE_MODEL_FILE = os.path.join(outputDir, 'model.tflite')
#     with open(self.TFLITE_MODEL_FILE, 'wb') as f:
#         f.write(tflite_model)
#     print(f"Generated tflite model in {outputDir}")
#

def representative_dataset_random():
    global _width, _height

    for _ in range(100):
        data = np.random.rand(_height, _width, 3) * 2 - 1 # -1 ~ 1
        out  = np.expand_dims(data, 0).astype(np.float32)
        yield [out]

# def representative_dataset():
#     dataset_list = tf.data.Dataset.list_files('road_data/images/train/*.jpg')
#     for i in range(100):
#         image = next(iter(dataset_list))
#         image = tf.io.read_file(image)
#         image = tf.io.decode_jpeg(image, channels=3)
#         image = tf.image.resize(image, [_height, _width])
#         image = tf.cast(image / 255., tf.float32)
#         image = tf.expand_dims(image, 0)
#         yield [image]

def representative_dataset_tfrecord():
    global _dataset
    cnt = 0
    for pp, d in enumerate(_dataset):
        # print(d['image/filename'].numpy(), d['image/width'].numpy(), d['image/height'].numpy())
        # print(d['image/source_id'].numpy(), d['image/format'].numpy())
        img = tf.image.decode_jpeg(d['image/encoded'][0], channels=3)
        np_img = np.array(img)
        width  = d['image/width'].numpy()
        height = d['image/height'].numpy()
        xmin = d['image/object/bbox/xmin'].values.numpy() * width
        ymin = d['image/object/bbox/ymin'].values.numpy() * height
        xmax = d['image/object/bbox/xmax'].values.numpy() * width
        ymax = d['image/object/bbox/ymax'].values.numpy() * height

        xmin = np.int32(xmin)
        xmax = np.int32(xmax)
        ymin = np.int32(ymin)
        ymax = np.int32(ymax)

        if cnt > 80:
            break

        iter = len(xmin)
        for j in range(iter):
            image = np_img[ymin[j]:ymax[j], xmin[j]:xmax[j], ]
            image = tf.image.resize(image, [_height, _width])
            image = (image - _INPUT_NORM_MEAN) / _INPUT_NORM_STD
            image = np.expand_dims(image, 0).astype(np.float32)
            cnt = cnt + 1
            yield [image]

feature_description={
           'image/height': tf.io.FixedLenFeature([], tf.int64),
           'image/width': tf.io.FixedLenFeature([], tf.int64),
           'image/filename': tf.io.FixedLenFeature([], tf.string),
           'image/source_id': tf.io.FixedLenFeature([], tf.string),
           'image/encoded': tf.io.FixedLenFeature([], tf.string),
           'image/format': tf.io.FixedLenFeature([], tf.string),
           'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
           'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
           'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
           'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
           'image/object/class/text': tf.io.VarLenFeature(tf.string),
           'image/object/class/label': tf.io.VarLenFeature(tf.int64)
           }

def parse_tfrecord(tfrecord):
    x = tf.io.parse_single_example(tfrecord, feature_description)
    return x


def write_tflite(saved_model_path, tflite_file_name, is_quantize, width, height):
    global _width, _height
    global _dataset

    split_ext = os.path.splitext(saved_model_path)
    if split_ext[1] == '.h5':
        print('keras')
        model = load_model(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path, signature_keys=['serving_default'])

    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    ]

    if is_quantize:
        _width   = width
        _height  = height

        record_file = "road_data/train.record"
        raw_dataset = tf.data.TFRecordDataset(record_file)
        _dataset = raw_dataset.map(parse_tfrecord)
        _dataset = _dataset.batch(1)

        converter.representative_dataset = representative_dataset_tfrecord
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
        converter.inference_input_type = tf.int8
        #converter.inference_output_type = tf.int8
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    tflite_out = converter.convert()
    open(tflite_file_name, "wb").write(tflite_out)


def write_meta_tflite(model_file, label_file, output_file):
    if not os.path.exists(model_file):
        print(f'file not found {model_file}')
        return

    if not os.path.exists(label_file):
        print(f'file not found {label_file}')
        return

    tmp_file = '.label.txt'
    category_index = label_map_util.create_category_index_from_labelmap(label_file)
    last = [*category_index.keys()][-1] + 1

    f = open(tmp_file, 'w')
    for class_id in range(1, last):
        if class_id not in category_index:
            f.write('???\n')
            continue
        name = category_index[class_id]['name']
        f.write(name + '\n')
    f.close()

    # names = os.path.splitext(model_file)
    # output_file = names[0] + '_metadata' + names[1]

    ObjectDetectorWriter = object_detector.MetadataWriter
    # Normalization parameters is required when reprocessing the image. It is
    # optional if the image pixel values are in range of [0, 255] and the input
    # tensor is quantized to uint8. See the introduction for normalization and
    # quantization parameters below for more details.
    # https://www.tensorflow.org/lite/models/convert/metadata#normalization_and_quantization_parameters)
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD  = 127.5

    # Create the metadata writer.
    writer = ObjectDetectorWriter.create_for_inference(
        writer_utils.load_file(model_file), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [tmp_file])

    # Verify the metadata generated by metadata writer.
    # print(writer.get_metadata_json())

    # Populate the metadata into the model.
    writer_utils.save_file(writer.populate(), output_file)
    os.remove(tmp_file)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            'usage : python tflite_cnv.py [in:saved model path] [in:pbtxt label file] [out:tflite file] (in:quantize input width) (in:quantize input height)')
    else:
        width, height = 0, 0
        is_quantize = False
        if len(sys.argv) >= 6:
            is_quantize = True
            width, height = int(sys.argv[4]), int(sys.argv[5])

        tmp_tflite_file = './.tmp.tflite'
        write_tflite(sys.argv[1], tmp_tflite_file, is_quantize, width, height)
        write_meta_tflite(tmp_tflite_file, sys.argv[2], sys.argv[3])
        os.remove(tmp_tflite_file)
