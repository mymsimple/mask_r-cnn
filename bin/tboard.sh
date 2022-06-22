if [ "$1" = "" ] || [ "$1" = "help" ]; then
   echo "Usage: tboard.sh <port> <#gpu_id>"
   echo "       tboard.sh stop"
   exit
fi
set -x
if [ "$1" = "stop" ]; then
   ps aux|grep tensorboard|grep mask_rcnn|awk '{print $2}'| xargs kill
   echo "[mask_rcnn]TensorBoard退出..."
   exit
fi

gpu=3

log_path=logs/
port=$1
echo "启动 tensorboard日志...."
echo "端口：$port，日志路径：$log_path"
CUDA_VISIBLE_DEVICES=$gpu nohup /root/py3/bin/tensorboard --port=$port --logdir=$log_path >./tboard.log 2>&1 &