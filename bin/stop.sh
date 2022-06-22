#ps aux|grep mask_rcnn|grep -v grep|awk '{print $1}'|xargs kill -9
ps -ef|grep mask_train|awk '{print $2}'|xargs kill -9