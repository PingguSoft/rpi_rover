import os
import sys
from object_detection.utils import label_map_util
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils


def write_meta_tflite(model_file, label_file):
    if not os.path.exists(model_file):
        print(f'file not found {model_file}')
        return
        
    if not os.path.exists(label_file):
        print(f'file not found {label_file}')
        return
    
    tmp_file = label_file + '.tmp'
    category_index = label_map_util.create_category_index_from_labelmap(label_file)
    last = [*category_index.keys()][-1] + 1
    
    f = open(tmp_file, 'w')
    for class_id in range(1, last):
        if class_id not in category_index:
            f.write('???\n')
            continue
        name = category_index[class_id]['name']
        f.write(name+'\n')
    f.close()

    
    model_buf = writer_utils.load_file(model_file)
    #names = os.path.splitext(model_file)
    output_file = model_file #names[0] + '_metadata' + names[1]

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
        model_buf, [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [tmp_file])

    # Verify the metadata generated by metadata writer.
    # print(writer.get_metadata_json())

    # Populate the metadata into the model.
    writer_utils.save_file(writer.populate(), output_file)
    os.remove(tmp_file)
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage : python metadata_writer.py [model file] [pbtxt label file]')
    else:
        write_meta_tflite(sys.argv[1], sys.argv[2])