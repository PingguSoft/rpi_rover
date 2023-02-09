#!/bin/bash
if [[ $# -lt 3 ]]; then
    echo "saved_model_2_tflite.sh [saved_model path] [pb.txt file] [tflite file name] (quant width) (quant height)"
    exit 2
fi

echo -e "\n"
echo -e "----------------------------------------------------------------"
echo ">>> 1.converting saved model to tflite"
echo -e "----------------------------------------------------------------"
python tflite_cnv_od.py $1 $2 $3 $4 $5


if [[ $# -gt 4 ]]; then
    echo -e "\n"
    echo -e "----------------------------------------------------------------"
    echo ">>> 2.converting tflite to edgetpu_tflite"
    echo -e "----------------------------------------------------------------"
    edgetpu_compiler -s $3

    filename="${3%%.*}"_edgetpu.tflite
    echo -e "\n"
    echo -e "----------------------------------------------------------------"
    echo ">>> 3.writing metadata to edgetpu_tflite "
    echo -e "----------------------------------------------------------------"
    python metadata_writer.py $filename $2
fi
echo -e "\n"
