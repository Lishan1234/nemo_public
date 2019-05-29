import os
import sys

num_blocks_ = [2, 4, 6, 8]
num_filters_ = [16, 24, 32, 48]

cmd = 'python setup_snpe.py --model_type edsr_v2 --num_reduced_filters 3 --num_blocks 8 --num_filters 64 --upsample_type transpose --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'
os.system(cmd)

cmd = 'python setup_snpe.py --model_type edsr_v2 --num_reduced_filters 3 --num_blocks 8 --num_filters 64 --upsample_type resize_bilinear --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'
os.system(cmd)

"""
#exp1
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
        cmd = 'python setup_snpe.py --num_blocks {} --num_filters {} --upsample_type transpose --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'.format(num_blocks, num_filters)
        #os.system(cmd)

#exp2
num_blocks_ = [4]
num_filters_ = [32]
num_reduced_filters_ = [4, 8]
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
        for num_reduced_filters in num_reduced_filters_:
            cmd = 'python setup_snpe.py --model_type edsr_v2 --num_reduced_filters {} --num_blocks {} --num_filters {} --upsample_type resize_bilinear --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'.format(num_reduced_filters, num_blocks, num_filters)
            os.system(cmd)

#exp3
num_blocks_ = [4]
num_filters_ = [16, 24, 32, 48]
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
        cmd = 'python setup_snpe.py --model_type edsr_v1  --num_blocks {} --num_filters {} --upsample_type transpose --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'.format(num_blocks, num_filters)
        #os.system(cmd)

#exp4
num_blocks = 4
num_filters = 32
cmd = 'python setup_snpe.py --model_type edsr_v4 --num_blocks {} --num_filters {} --upsample_type transpose --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'.format(num_blocks, num_filters)
#os.system(cmd)
cmd = 'python setup_snpe.py --model_type edsr_v4 --num_blocks {} --num_filters {} --upsample_type subpixel --hwc 240,426,3 --train_data news --data_type 60_0.5 --lr 240 --snpe_project_root "../../../snpe" --snpe_tensorflow_root "../../../../tensorflow"'.format(num_blocks, num_filters)
#os.system(cmd)
"""
