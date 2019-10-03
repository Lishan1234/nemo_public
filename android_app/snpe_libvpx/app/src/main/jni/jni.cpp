#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <cerrno>
#include <cstdio>

#include <fstream>
#include <iostream>

//#include "snpe/examples/NativeCpp/SampleCode/jni/main.hpp"
//#include "main.hpp"

#include <chrono>
#include <thread>
#include <cmath>
#include <stdbool.h>

#include "main.hpp"

extern "C" {


	JNIEXPORT jlong JNICALL Java_com_example_snpe_1libvpx_MainActivity_function
	(JNIEnv *, jobject, jint test)
	{
		//Print test
		__android_log_print(ANDROID_LOG_ERROR, "JNITAG", "hi jni fine%d",test);

		//Default arguments
		const char * name = "";
	    const char * buffer_type = "ITENSOR";
		const char * input_path = "/sdcard/SNPEData/target_abs_raw_list.txt";
		const char * output_path = "/sdcard/SNPEData/output";
	    const char * dlc_path = "/sdcard/SNPEData/EDSR_transpose_B8_F48_S4.dlc";

	    //Call main
        int ret = fake_main(name, buffer_type, dlc_path, input_path, output_path);

	    return ret;
	}
}
