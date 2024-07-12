
CKPT='checkpoint.pth'

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python evaluate_error.py --config_file config/flir/zhang-bihome-lr-1e-2-test.yaml \
                                                         --ckpt $CKPT \