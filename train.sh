#!/bin/sh
cd /home/lss/caffe/unet/unet-caffe/
pwd
export PYTHONPATH=/home/lss/caffe/unet/unet-caffe:$PYTHONPATH
caffe.bin train --solver=solver.prototxt
