//
// Created by intern on 19. 5. 26.
//

#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <cerrno>
#include <cstdio>

#include <fstream>
#include <iostream>

//#include "snpe/examples/NativeCpp/SampleCode/jni/main.hpp"
#include "main.hpp"

#include <chrono>
#include <thread>
#include <cmath>
#include <stdbool.h>



extern "C" {



JNIEXPORT jlong JNICALL Java_com_example_battery_1profiler_ExoSNPE_function1
	(JNIEnv *env, jobject job, jint model, jint minutes, jint fps)
{
	//snpe-sample -b ITENSOR -d inception_v3_quantized.dlc -i target_raw_list.txt -o output_sample
	//b = type of buffers to use
	//d = path to DL container of network
	//i = path to file listeing inputs for network
	//o = path to store output results

    const char * name = "";
    const char * b = "ITENSOR";
	const char * i = "/sdcard/SNPEData/target_abs_raw_list.txt";
	const char * o = "/sdcard/SNPEData/output";
    const char * d;

    //b2 = lq
    if(model == 1){
	    d = "/sdcard/SNPEData/EDSR_transpose_B2_F16_S4.dlc";
	}

    //b8 = hq
	else{
	    d = "/sdcard/SNPEData/EDSR_transpose_B8_F48_S4.dlc";
	}

    __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "hi jni fine%d, %d fps",minutes, fps);

	long ret = fake_main(name,b,d,i,o,minutes,fps);

//	struct timeval now,begin;
//	gettimeofday(&now, NULL);
//	__android_log_print(ANDROID_LOG_ERROR, "JNITAG", "Testing %ld %ld",begin.tv_sec,begin.tv_usec);
//	long ret = 1;

	return ret;
}
}
