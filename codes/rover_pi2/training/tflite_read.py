from tflite_support import metadata

displayer = metadata.MetadataDisplayer.with_model_file("ssd_mobilenet_v2_coco17_q8.tflite")
print("Metadata populated:")
print(displayer.get_metadata_json())

print("Associated file(s) populated:")
for file_name in displayer.get_packed_associated_file_list():
  print("file name: ", file_name)
  print("file content:")
  print(displayer.get_associated_file_buffer(file_name))
