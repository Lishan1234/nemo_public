//==============================================================================
//
//  Copyright (c) 2015-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <ctime>
#include <unistd.h>

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
#include "CreateGLBuffer.hpp"
#endif

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "DiagLog/IDiagLog.hpp"

#include <android/log.h>
#include <chrono>
#include <thread>
#include <stdbool.h>
#include <cmath>

const int FAILURE = 1;
const int SUCCESS = 0;


bool within_one_second(struct timeval * before, struct timeval * after){
    long long before_msec = (*before).tv_usec/1000 + (*before).tv_sec*1000;
    long long after_msec = (*after).tv_usec/1000 + (*after).tv_sec*1000;

    if(llabs(after_msec- before_msec) <= 1000ll){
        return true;
    }else{
        return false;
    }
}

bool within_sixty_seconds(struct timeval * before, struct timeval * after,int minutes){
    long int before_msec = (*before).tv_usec/1000 + (*before).tv_sec*1000;
    long int after_msec = (*after).tv_usec/1000 + (*after).tv_sec*1000;
    if(llabs(after_msec - before_msec) <= 60000l * minutes){
        return true;
    }else{
        return false;
    }
}


long fake_main(const char * name, const char * b, const char * d, const char * i, const char * o,int minutes,int fps)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
    bool execStatus = false;
    bool usingInitCaching = false;

    long count = 0;

    // Process command line arguments
    int opt = 0;
    bufferTypeStr = b;
    dlc = d;
    inputFile = i;
    OutputDir = o;

    __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "start");

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        // std::exit(FAILURE);
        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "files INVALID");
        return 1;    }

    // Check if given buffer type is valid
    int bufferType;
    if (bufferTypeStr == "USERBUFFER_FLOAT")
    {
        bufferType = USERBUFFER_FLOAT;
    }
    else if (bufferTypeStr == "USERBUFFER_TF8")
    {
        bufferType = USERBUFFER_TF8;
    }
    else if (bufferTypeStr == "ITENSOR")
    {
        bufferType = ITENSOR;
    }
    else
    {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        // std::exit(FAILURE);
        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "buffer type INVALID");
        return 1;
    }

    //Check if given user buffer source type is valid
    int userBufferSourceType;
    // CPUBUFFER / GLBUFFER supported only for USERBUFFER_FLOAT
    if (bufferType == USERBUFFER_FLOAT)
    {
        if( userBufferSourceStr == "CPUBUFFER" )
        {
            userBufferSourceType = CPUBUFFER;
        }
        else if( userBufferSourceStr == "GLBUFFER" )
        {
#ifndef ANDROID
            std::cout << "GLBUFFER mode is only supported on Android OS" << std::endl;
            // std::exit(FAILURE);
            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "GLBuffer INVALID");
            return 1;
#endif
            userBufferSourceType = GLBUFFER;
        }
        else
        {
            std::cout
                  << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
            // std::exit(FAILURE);
                  __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "user buffer INVALID");
                  return 1;
        }
    }

    // Open the DL container that contains the network to execute.
    // Create an instance of the SNPE network from the now opened container.
    // The factory functions provided by SNPE allow for the specification
    // of which layers of the network should be returned as output and also
    // if the network should be run on the CPU or GPU.
    // The runtime availability API allows for runtime support to be queried.
    // If a selected runtime is not available, we will issue a warning and continue,
    // expecting the invalid configuration to be caught at SNPE network creation.
    zdl::DlSystem::UDLFactoryFunc udlFunc = sample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie

    runtime = checkRuntime(runtime);
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       // std::exit(FAILURE);
       __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "opening container INVALID");
       return 1;
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;
#ifdef ANDROID
    CreateGLBuffer* glBuffer = nullptr;
    if (userBufferSourceType == GLBUFFER) {
        if(!checkGLCLInteropSupport()) {
            std::cerr << "Failed to get gl cl shared library" << std::endl;
            // std::exit(1);
            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "gl cl INVALID");
            return 1;
        }
        glBuffer = new CreateGLBuffer();
        glBuffer->setGPUPlatformConfig(platformConfig);
    }
#endif

    snpe = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);
    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       // std::exit(FAILURE);
       __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "building snpe object INVALID");
       return 1;
    }
    if (usingInitCaching)
    {
       if (container->save(dlc))
       {
          std::cout << "Saved container into archive successfully" << std::endl;
       }
       else
       {
          std::cout << "Failed to save container into archive" << std::endl;
       }
    }

    // Configure logging output and start logging. The snpe-diagview
    // executable can be used to read the content of this diagnostics file
    auto logger_opt = snpe->getDiagLogInterface();
    if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
    auto logger = *logger_opt;
    auto opts = logger->getOptions();

    opts.LogFileDirectory = OutputDir;
    if(!logger->setOptions(opts)) {
        std::cerr << "Failed to set options" << std::endl;
        // std::exit(FAILURE);
        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "failed to set options");
        return 1;
    }
    if (!logger->start()) {
        std::cerr << "Failed to start logger" << std::endl;
        //std::exit(FAILURE);
        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "failed to start logger");
        return 1;
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
#ifdef ANDROID
    size_t bufSize = 0;
    if (userBufferSourceType == GLBUFFER) {
        if(batchSize > 1) {
            std::cerr << "GL buffer source mode does not support batchsize larger than 1" << std::endl;
            // std::exit(1);
            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "gl buffer som eerror");
            return 1;
        }
        bufSize = calcSizeFromDims(tensorShape.getDimensions(), tensorShape.rank(), sizeof(float));
    }
#endif
    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);

    // Load contents of input file batches ino a SNPE tensor or user buffer,
    // user buffer include cpu buffer and OpenGL buffer,
    // execute the network with the input and save each of the returned output to a file.
    if(useUserSuppliedBuffers)
    {
       // SNPE allows its input and output buffers that are fed to the network
       // to come from user-backed buffers. First, SNPE buffers are created from
       // user-backed storage. These SNPE buffers are then supplied to the network
       // and the results are stored in user-backed output buffers. This allows for
       // reusing the same buffers for multiple inputs and outputs.
       zdl::DlSystem::UserBufferMap inputMap, outputMap;
       std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

       if( bufferType == USERBUFFER_TF8 )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, true);

          std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
          createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, true);

          for( size_t i = 0; i < inputs.size(); i++ )
          {
             // Load input user buffer(s) with values from file(s)
             if( batchSize > 1 )
                std::cout << "Batch " << i << ":" << std::endl;
             loadInputUserBufferTf8(applicationInputBuffers, snpe, inputs[i], inputMap);
             // Execute the input buffer map on the model with SNPE
             execStatus = snpe->execute(inputMap, outputMap);
             // Save the execution results only if successful
             if (execStatus == true)
             {
                saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, true);
             }
             else
             {
                std::cerr << "Error while executing the network." << std::endl;
             }
          }
       }
       else if( bufferType == USERBUFFER_FLOAT )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, false);

             for( size_t i = 0; i < inputs.size(); i++ )
             {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBufferFloat(applicationInputBuffers, snpe, inputs[i]);
                // Execute the input buffer map on the model with SNPE
                execStatus = snpe->execute(inputMap, outputMap);
                // Save the execution results only if successful
                if (execStatus == true)
                {
                   saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, false);
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }
             }
          }
#ifdef ANDROID
            if(userBufferSourceType  == GLBUFFER) {
                std::unordered_map<std::string, GLuint> applicationInputBuffers;
                createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe);
                GLuint glBuffers = 0;
                for(size_t i = 0; i < inputs.size(); i++) {
                    // Load input GL buffer(s) with values from file(s)
                    glBuffers = glBuffer->convertImage2GLBuffer(inputs[i], bufSize);
                    loadInputUserBuffer(applicationInputBuffers, snpe, glBuffers);
                    // Execute the input buffer map on the model with SNPE
                    execStatus =  snpe->execute(inputMap, outputMap);
                    // Save the execution results only if successful
                    if (execStatus == true) {
                        saveOutput(outputMap, applicationOutputBuffers, OutputDir, i*batchSize, batchSize, false);
                    }
                    else
                    {
                        std::cerr << "Error while executing the network." << std::endl;
                    }
                    // Release the GL buffer(s)
                    glDeleteBuffers(1, &glBuffers);
                }
            }
#endif
       }
    }
    else if(bufferType == ITENSOR)
    {
        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;

        //input stuff
        double seconds = minutes*60;

        size_t i = 0;

        //Save current time
        long int diff = 0;
        struct timeval now,begin;
        gettimeofday(&now, NULL);
        begin = now;

        //let's say we have a given frames per second. 
        //once we reached this number in a second, we should sleep for the remaining time.

        //fps stuff
        int array_index = 0;
        struct timeval array[minutes*60 + 1];
        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "begin %ld",begin.tv_sec);
        for(int i = 1; i <=minutes*60+1; i++){
            struct timeval temp;
            temp.tv_sec = begin.tv_sec + i;
            temp.tv_usec = begin.tv_usec;
            array[i-1]=temp;
            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "array %d %ld",i,array[i-1].tv_sec);
        }

        int target_fps = fps;
        int fps_count = 0;
        struct timeval fps_now;
        fps_now = now;

        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "Start, s = %ld ms= %ld",0,begin.tv_usec);

        while(within_sixty_seconds(&begin,&now,minutes)){
            //Print time values
            diff = now.tv_sec-begin.tv_sec;

            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "Inference begin, diff = %d",diff);

            //now should always be array[array_index-1] ≤ now < array[array_index]
            struct timeval current = array[array_index];
            long int temp = (1000 - now.tv_usec/1000) + current.tv_usec/1000;
            if( (now.tv_sec == current.tv_sec && now.tv_usec < current.tv_usec) || (now.tv_sec < current.tv_sec && temp <= 1000)){//same second
                if(fps_count == target_fps){
                    //sleep
                    long int sleep = 0;
                    if(now.tv_sec == current.tv_sec){
                        sleep += (current.tv_usec - now.tv_usec)/1000;
                    }else{
                        sleep += 1000 - now.tv_usec/1000;//milliseconds
                        sleep += current.tv_usec/1000;
                    }
                    __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "sleep time: %ld",sleep);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep+1));
//                    usleep(sleep*1000);

                    gettimeofday(&now, NULL);
                    continue;
                }
                __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "no sleep");
            }else{//next second
                array_index++;
                fps_count = 0;
                __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "next sec");
            }


            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;

            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[i]);
            if (inputTensor == nullptr){
                return -1;
            }

            // Execute the input tensor on the model with SNPE
            __android_log_print(ANDROID_LOG_ERROR,"JNITAG","Executing...");

            execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
            if(execStatus ==false){
                __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "Error while executing network");
                return -1;
            }

            //update time and count
            gettimeofday(&now, NULL);
            count++;
            fps_count++;
            __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "Inference end");
        }
        __android_log_print(ANDROID_LOG_ERROR,"JNITAG","c++ code done, inference count is %ld", count);
    }
    return count;
}


