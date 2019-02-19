This MATLAB packages provides quick access to call HEVC encoder to encode a given YUV video in the STANDARD testing video set and load the dumped information about the encoded bit stream into MATLAB. The dumped information includes: PU structure, TU strucuture, motion vectors, motion compensated blocks, residues, final intra blocks (prediction + residue).

--------------------------------------------------------------------------
- Setup (Optional) 
--------------------------------------------------------------------------
1. (Only needed if you change the included HM software for dumping more info) Recompile the HM software: go to the directory containing the folder hm as a direct sub folder and run the following command in the command line:

make -C hm/build/linux release

This command is also included in the batch file: compile_hm.sh. So you can also run it if you make no changes to the directory structure.

2. (needed only if you put the included HM software somewhere else). Set the paths to HM software in make_encoding_param.m

--------------------------------------------------------------------------
- Run the package 
--------------------------------------------------------------------------
The software contains multiple functions to encode the sequence, decode the sequence, and to parse the dumped information. Please refer to unit_test_parse_all_saved_info.m for how to use the package.

--------------------------------------------------------------------------
- Notes 
--------------------------------------------------------------------------
1. The encoding function is specifically designed for the naming style of the standard testing videos, like BQSquare_416x240.yuv, with the format seqname_widthxheight.yuv. Otherwise you need to modify the code to locate the file and you need to manually provide the height and width.

2. For intra blocks, the returned reconstructed block is the final one: deblocking + SAO on (prediction + residue). For inter blocks, the returned reconstructed block is just the mc prediction + residue. 

3. If you need raw intra prediction (without residue, and any post processing), you can read them from prefilt.yuv in the enc_files subfolder. This is not dumped because it is not used in my research. The non-intra blocks in prefilt.yuv are not meaningful so you can ignore them. 

4. The PU structure is included in other_info.PU. The TU structure is in other_info.TU. The motion vectors are also included in other_info.

5. The intermediate results, including the dumped information, the reconstructed yuv files and the encoded bit streams are placed in the subfolder enc_files. The code does NOT remove them afterwards.

6. The code supports the encoding of arbitrary number of frames starting from the first frame. You can set it to empty to encode the whole stream, which will be very slow, and the dumped file will be very large.

--------------------------------------------------------------------------
- Contact
--------------------------------------------------------------------------
Author: Mehul Tikekar for HM dumping, Zhengdong Zhang for MATLAB interface
Email for Questions: zhangzdfaint@gmail.com

