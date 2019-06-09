import os
import sys

#num_blocks_ = [2, 4, 6, 8]
#num_filters_ = [16, 24, 32, 48]
num_blocks_ = [8]
num_filters_ = [32]

json_file_name = 'news_EDSR_v2_transpose_B8_F64_RF3_S4.json'
cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
os.system(cmd)
json_file_name = 'news_EDSR_v2_resize_bilinear_B8_F64_RF3_S4.json'
cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
os.system(cmd)

"""
#exp1
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
        json_file_name = 'news_EDSR_transpose_B{}_F{}_S4.json'.format(num_blocks, num_filters)
        cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
        #os.system(cmd)

#exp2
num_blocks_ = [4]
num_filters_ = [32]
num_reduced_filters_ = [4, 8]
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
        for num_reduced_filters in num_reduced_filters_:
            json_file_name = 'news_EDSR_v2_resize_bilinear_B{}_F{}_RF{}_S4.json'.format(num_blocks, num_filters, num_reduced_filters)
            cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
            #os.system(cmd)

#exp3
num_blocks_ = [4]
num_filters_ = [16, 24, 32, 48]
for num_blocks in num_blocks_:
    for num_filters in num_filters_:
            json_file_name = 'news_EDSR_v1_transpose_B{}_F{}_S4.json'.format(num_blocks, num_filters)
            cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
            #os.system(cmd)

json_file_name = 'news_EDSR_v4_transpose_B4_F32_S4.json'
cmd = 'python snpe_bench.py -c {} -a'.format(json_file_name)
os.system(cmd)
"""
