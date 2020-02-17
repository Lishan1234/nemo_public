#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <android/log.h>
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#ifdef ANDROID
#include <GLES2/gl2.h>
#include <jni.h>
#include <iosfwd>
#include "CreateGLBuffer.hpp"
#endif



const int FAILURE = 1;
const int SUCCESS = 0;

int fake_main(int, const char *, const char *, const char *, const char *, bool doLog);


extern "C" {

JNIEXPORT jlong JNICALL Java_com_example_exo_1snpe_1battery_MainActivity_jniFunction
        (JNIEnv * env, jobject obj, jint minutes, jstring dlc, jstring input, jstring output, jstring log, jboolean doLog)
{

    const char * _dlc = env->GetStringUTFChars(dlc, NULL);
    const char * _input = env->GetStringUTFChars(input, NULL);
    const char * _output = env->GetStringUTFChars(output, NULL);
    const bool _doLog = (bool) doLog;

    if(doLog){
        const char * _log = env->GetStringUTFChars(log, NULL);
        fake_main(minutes, _dlc, _input,_output,_log, doLog);
    }
    else{
        fake_main(minutes, _dlc, _input, _output, NULL,doLog);
    }

    return 1;
}

bool time_passed(struct timeval * before, struct timeval *after, int minutes){
    long int before_msec = (*before).tv_usec/1000 + (*before).tv_sec*1000;
    long int after_msec = (*after).tv_usec/1000 + (*after).tv_sec*1000;
    if(llabs(after_msec-before_msec) <= 60000);

    return (llabs(after_msec-before_msec) > 60000 * minutes);
}


}

int fake_main(int minutes, const char * d, const char * i, const char * o, const char * l, bool doLog){

    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
    enum {CPUBUFFER, GLBUFFER};

    //command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
    static zdl::DlSystem::RuntimeList runtimeList;
    bool runtimeSpecified = false;
    bool execStatus = false;
    bool usingInitCaching = false;

    //Process command line arguments
    int opt = 0;
    dlc = d;
    inputFile = i;
    OutputDir = o;
    runtimeSpecified = true;
    std::string buffer ="";
    static std::string log = "";
    if(doLog){
        log = l;
    }



    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        return EXIT_FAILURE;
    }


    // Check if given buffer type is valid
    int bufferType = ITENSOR;


    //Check if both runtimelist and runtime are passed in
    if(runtimeSpecified && runtimeList.empty() == false)
    {
        std::cout << "Invalid option cannot mix runtime order -l with runtime -r " << std::endl;
        std::exit(FAILURE);
    }


    // Open the DL container that contains the network to execute.
    // Create an instance of the SNPE network from the now opened container.
    // The factory functions provided by SNPE allow for the specification
    // of which layers of the network should be returned as output and also
    // if the network should be run on the CPU or GPU.
    // The runtime availability API allows for runtime support to be queried.
    // If a selected runtime is not available, we will issue a warning and continue,
    // expecting the invalid configuration to be caught at SNPE network creation.
    zdl::DlSystem::UDLFactoryFunc udlFunc = UdlExample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie

//    if(runtimeSpecified)
//    {
//        runtime = checkRuntime(runtime);
//    }

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       return EXIT_FAILURE;
    }

    bool useUserSuppliedBuffers = false;

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;

    snpe = setBuilderOptions(container, runtime, runtimeList, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);
    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       return EXIT_FAILURE;
    }


    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);


    if(bufferType == ITENSOR)
    {
        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;

        // Load input/output buffers with ITensor
        std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[0]);
        if(!inputTensor)
        {
            return EXIT_FAILURE;
        }

        struct timeval now, begin;
        gettimeofday(&now, NULL);
        begin = now;


        struct timeval start, end;


        while(!time_passed(&begin,&now,minutes)){
            //execute snpe
            __android_log_print(6,"TAG","execute");


            gettimeofday(&start, NULL);
            execStatus = snpe->execute(inputTensor.get(),outputTensorMap);
            if(execStatus == true){
//                if(!saveOutput(outputTensorMap, OutputDir, 0, batchSize))
//                {
//                    __android_log_print(6,"TAG", "save output failure");
//                    return EXIT_FAILURE;
//                }
            }else{
                __android_log_print(6,"TAG","execution failure");
            }
            __android_log_print(6,"TAG","execution done");

            gettimeofday(&end,NULL);
            long int start_msec = start.tv_usec/1000 + start.tv_sec*1000;
            long int end_msec = end.tv_usec/1000 + end.tv_sec*1000;
            buffer += std::to_string(end_msec - start_msec);
            buffer += "\n";

            //update time
            gettimeofday(&now,NULL);
        }


        //write all latencies
        if(doLog){
            std::string filePath = log + "/latency.txt";
            __android_log_print(6,"TAG", "log%s", filePath.c_str());

            std::ofstream writeFile(filePath);
            writeFile << buffer;
            writeFile.close();
        }


    }


    __android_log_print(6,"TAG","time loop complete");
    // Freeing of snpe object
    snpe.reset();
    return SUCCESS;

}