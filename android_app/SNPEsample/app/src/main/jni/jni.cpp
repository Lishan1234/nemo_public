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
extern "C" {


JNIEXPORT jlong JNICALL Java_com_example_snpesample_SNPEActivity_function1
	(JNIEnv *env, jobject job, jint model, jint minutes)
{
	//snpe-sample -b ITENSOR -d inception_v3_quantized.dlc -i target_raw_list.txt -o output_sample
	//b = type of buffers to use
	//d = path to DL container of network
	//i = path to file listeing inputs for network
	//o = path to store output results

	// char * name = "snpe-sample";
	// char * b = "ITENSOR";
	// char * d = "inception_v3_quantized.dlc";
	// char * i = "target_raw_list.txt";
	// char * o = "output_sample";

	//SNPE sample with original directories
	//	const char * name = "";//doens't matter
	//	const char * b = "ITENSOR";
	// const char * d = "/data/local/tmp/inception_v3/inception_v3_quantized.dlc";
	// const char * i = "/data/local/tmp/inception_v3/target_raw_list.txt";
	// const char * o = "/data/local/tmp/snpesample/output_sample";

	//SNPE sample with change directories
//	const char * name = "";//doens't matter
//	const char * b = "ITENSOR";
//	const char * d = "/storage/emulated/0/Android/data/com.example.snpesample/inception_v3/inception_v3_quantized.dlc";
//	const char * i = "/storage/emulated/0/Android/data/com.example.snpesample/inception_v3/target_raw_list.txt";
//	const char * o = "/storage/emulated/0/Android/data/com.example.snpesample/files/output_sample";


    //hyunho model
    const char * name = "";
    const char * b = "ITENSOR";
    const char * d;
    const char * i = "/storage/emulated/0/Android/data/com.example.snpesample/270p/target_abs_raw_list.txt";
    const char * o = "/storage/emulated/0/Android/data/com.example.snpesample/270p/output";
    if(model == 1){
	    //b2
        __android_log_print(ANDROID_LOG_ERROR, "HI", "hi jni fine");
        d = "/storage/emulated/0/Android/data/com.example.snpesample/270p/EDSR_transpose_B2_F16_S4.dlc";
	}
	else{
	    //should be only 4 for now
        d = "/storage/emulated/0/Android/data/com.example.snpesample/270p/EDSR_transpose_B8_F48_S4.dlc";
	}



	i = "/sdcard/mobinas/target_abs_raw_list.txt";
	o = "/sdcard/mobinas/output";
	if(model == 1){
	    d = "/sdcard/mobinas/EDSR_transpose_B2_F16_S4.dlc";
	}
	else{
	    d = "/sdcard/mobinas/EDSR_transpose_B8_F48_S4.dlc";
	}


    __android_log_print(ANDROID_LOG_ERROR, "HI", "hi jni fine%d",minutes);


	long ret = fake_main(name,b,d,i,o,minutes);
	// test(name,b,d,i,o);
	return ret;
}

}