This package provides a basic MATLAB implementation of FAST, the framework to accelerate super-resolution on compressed videos. The code demonstrates how to encode a video from YUV format to H.265/HEVC format with the provided utility, and how to use the same utility to extract the syntax elements including block structures, residuals and motion compensation.

The package only provides the basic functionality of the FAST framework. More advanced features including enabling and disabling FAST by thresholding the accumulated Laplacian of Residual are not included. 

The code works on Windows, Linux and Mac.

-------------------------------------------------------------------
- Run the code on the provided data with SRCNN
-------------------------------------------------------------------
The code includes a CPU version of SRCNN[1], and the first 16 frames of a 720P sequence 'calendar' from the Consumer Digital Video Library. You can run /start_fast_sr.m and see how FAST + SRCNN works on this sequence.

-------------------------------------------------------------------
- Run your SR algorithm on your input video
-------------------------------------------------------------------
1. Prepare the video

The video needs to be in YUV 4:2:0 format. The name of the video should be SequenceName_WidthxHeight.yuv. A customized configuration file for the encoder needs to be created for the input sequence. You can modify any of the sequence file from /utils/matlab_call_hevc_tools/hm/cfg/per-sequence to create the customized configuration file. The file should be placed in the same folder.

The provided code only works in one Group-Of-Picture (GOP). So you should only provide the frames that you are interested in (16 frames in the provided data).

2. Write a wrapper for the SR algorithms

The framework calls the SR algorithm with a predefined wrapper format. Suppose your SR algorithm is called X, then you need to provide SRload_X.m that loads the data needed for your SR algorithms, and SR_X.m that calls the SR algorithm. Please refer to /utils/SRCNN/SRload_CNN.m, and /utils/SRCNN/SR_CNN.m for the format of the wrapper.

3. Modify start_fast_sr to run it!

-------------------------------------------------------------------
- Contact
-------------------------------------------------------------------
Zhengdong Zhang, zhangzd@mit.edu

-------------------------------------------------------------------
- References
-------------------------------------------------------------------
[1] SRCNN. http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
[2] CDVL. http://www.cdvl.org/