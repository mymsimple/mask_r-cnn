echo "开始训练"

Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0

log_dir="logs"

#GPU=-1
#
#if [ "$1" != "" ]; then
#    echo "选择GPU$1"
#    GPU=$1
#fi

if [ ! -d "$log_dir" ]; then
        mkdir $log_dir
fi

if [ "$1" == "console" ];then
   python -m mask_train
exit
fi

#python -m mask_train
#exit

nohup \
python -m mask_train \
>> ./logs/console_$Date.log 2>&1 &
echo "启动完毕,在logs下查看日志！"