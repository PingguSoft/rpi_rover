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

def representative_dataset():
    global _width, _height

    for _ in range(100):
        data = np.random.rand(_height, _width, 3)
        out  = np.expand_dims(data, axis=0).astype(np.float32)
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

# def representative_dataset():
#     global _dataset
#     cnt = 0
#     for pp, d in enumerate(_dataset):
#         # print(d['image/filename'].numpy(), d['image/width'].numpy(), d['image/height'].numpy())
#         # print(d['image/source_id'].numpy(), d['image/format'].numpy())
#         img = tf.image.decode_jpeg(d['image/encoded'][0], channels=3)
#         np_img = np.array(img)
#         width  = d['image/width'].numpy()
#         height = d['image/height'].numpy()
#         xmin = d['image/object/bbox/xmin'].values.numpy() * width
#         ymin = d['image/object/bbox/ymin'].values.numpy() * height
#         xmax = d['image/object/bbox/xmax'].values.numpy() * width
#         ymax = d['image/object/bbox/ymax'].values.numpy() * height
#
#         xmin = np.int32(xmin)
#         xmax = np.int32(xmax)
#         ymin = np.int32(ymin)
#         ymax = np.int32(ymax)
#
#         if cnt > 50:
#             break
#
#         iter = len(xmin)
#         for j in range(iter):
#             image = np_img[ymin[j]:ymax[j], xmin[j]:xmax[j], ]
#             image = tf.image.resize(image, [_height, _width])
#             image = tf.cast(image / 255., tf.float32)
#             image = tf.expand_dims(image, 0)
#             cnt = cnt + 1
#             yield [image]

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
        model = load_model(saved_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path, signature_keys=['serving_default'])

    # converter.allow_custom_ops = True
    # converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        # tf.lite.OpsSet.SELECT_TF_OPS,    # enable TensorFlow ops.
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    ]

    if is_quantize:
        _width = width
        _height = height

        # record_file = "road_data/train.record"
        # raw_dataset = tf.data.TFRecordDataset(record_file)
        # _dataset = raw_dataset.map(parse_tfrecord)
        # _dataset = _dataset.batch(1)

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
        converter.inference_input_type = tf.uint8  # or tf.uint8
        # converter.inference_output_type = tf.uint8  # or tf.uint8
    else:
        # converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
        # converter.inference_input_type = tf.float32
        # converter.inference_output_type = tf.float32
        pass

    tflite_out = converter.convert()
    open(tflite_file_name, "wb").write(tflite_out)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            'usage : python tflite_cnv.py [in:saved model path] [out:tflite file] (in:quantize input width) (in:quantize input height)')
    else:
        width, height = 0, 0
        is_quantize = False
        if len(sys.argv) >= 5:
            is_quantize = True
            width, height = int(sys.argv[3]), int(sys.argv[4])
        print(f'{is_quantize=}, {width=}, {height=}')
        write_tflite(sys.argv[1], sys.argv[2], is_quantize, width, height)
