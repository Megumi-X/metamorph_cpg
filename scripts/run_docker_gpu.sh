#!/bin/bash
# Launch an experiment using the docker gpu image
# Inside the metamorph folder run the following cmd:
# Usage: . scripts/run_docker_gpu.sh python metamorph/<file>.py

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

USER_ID=`id -u`
MOUNT_DIR='/home/xiongxiaoyu/metamorph/output'

docker run --gpus all -d --network host --ipc=host \
    --name metamorph \
    --restart unless-stopped \
    -v ${MOUNT_DIR}:/user/metamorph/output \
    metamorph \
    tail -f /dev/null