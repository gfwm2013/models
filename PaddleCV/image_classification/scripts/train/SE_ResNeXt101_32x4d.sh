##Training details
#GPU: NVIDIA® Tesla® P40 8cards 120epochs 566h
export CUDA_VISIBLE_DEVICES=4
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

#export FLAGS_conv_workspace_size_limit=4000 #MB
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_cudnn_batchnorm_spatial_persistent=1

DATA_DIR="/ssd3/datasets/ILSVRC2012"
DATA_FORMAT="NHWC"
#DATA_FORMAT="NCHW"

USE_FP16=false #whether to use float16
USE_DALI=true
USE_ADDTO=true

if ${USE_ADDTO} ;then
    export FLAGS_max_inplace_grad_add=8
fi

if ${USE_DALI}; then
    export FLAGS_fraction_of_gpu_memory_to_use=0.8
fi

python3 train.py \
       --model=SE_ResNeXt101_32x4d \
       --data_dir=${DATA_DIR} \
       --batch_size=32 \
       --total_images=1281167 \
       --image_shape 4 224 224 \
       --class_dim=1000 \
       --print_step=10 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay \
       --use_fp16=${USE_FP16} \
       --scale_loss=128.0 \
       --use_dynamic_loss_scaling=true \
       --data_format=${DATA_FORMAT} \
       --fuse_elewise_add_act_ops=true \
       --fuse_bn_act_ops=true \
       --fuse_bn_add_act_ops=true \
       --enable_addto=${USE_ADDTO} \
       --validate=true \
       --is_profiler=false \
       --profiler_path=profile/ \
       --reader_thread=10 \
       --reader_buf_size=4000 \
       --use_dali=${USE_DALI} \
       --lr=0.1 \
       --l2_decay=1.5e-5

##SE_ResNeXt101_32x4d:
#python3 train.py \
#        --model=SE_ResNeXt101_32x4d \
#        --data_dir=${DATA_DIR} \
#        --data_format=${DATA_FORMAT} \
#        --image_shape 3 224 224 \
#        --batch_size=64 \
#        --lr_strategy=cosine_decay \
#        --model_save_dir=output/ \
#        --lr=0.1 \
#        --num_epochs=200 \
#        --l2_decay=1.5e-5
