
#!/bin/bash

weight_path=models/zx_table/mask_rcnn_coco_0079.h5
output_path=models/pb_zx_table

echo "要转换的模型："  $weight_path
echo "转换后的路径："  $output_path

read -r -p "是否确认路径? 如果需要修改请选n。 [y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
		;;

    [nN][oO]|[nN])
      echo "No"
      read -r -p "请输入转换前模型：" weight_path
      read -r -p "请输出转换后路径：" output_path

      ;;
    *)
		echo "Invalid input..."
		exit 1
		;;
esac

set -x
python -m utils.convert_model -w $weight_path -o $output_path

#if [ $var == "ali" ] || [ $var == "huawei" ];then
#  python -m ocr_label -t $var
#else
#  echo "尚未完善"
#fi