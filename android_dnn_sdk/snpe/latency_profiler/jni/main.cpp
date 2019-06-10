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
//#include <ctime>
#include <chrono>

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

const int FAILURE = 1;
const int SUCCESS = 0;

//TODO: check whehter ITENSOR, CPUBUFFER can be improved

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    bool execStatus = false;
    bool usingInitCaching = false;
    int numIter;

    // Process command line arguments
    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:b:s:r:z:c:n:")) != -1)
    {
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C++ API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -d  <FILE>   Path to the DL container containing the network.\n"
                        << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "  -o  <PATH>   Path to directory to store output results.\n"
                        << "  -n  <NUMBER>  Number of iterations for measurement.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -b  <TYPE>   Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR] (" << bufferTypeStr << " is default).\n"
                        << "  -r  <RUNTIME> The runtime to be used [gpu, dsp, cpu] (cpu is default). \n"
                        << "  -z  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. \n"
#ifdef ANDROID
                        << "  -s  <TYPE>   Source of user buffers to use [GLBUFFER, CPUBUFFER] (" << userBufferSourceStr << " is default).\n"
                        << "               GL buffer is only supported on Android OS.\n"
#endif
                        << "  -c           Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << std::endl;

                std::exit(SUCCESS);
            case 'i':
                inputFile = optarg;
                break;
            case 'd':
                dlc = optarg;
                break;
            case 'o':
                OutputDir = optarg;
                break;
            case 'b':
                bufferTypeStr = optarg;
                break;
            case 's':
                userBufferSourceStr = optarg;
                break;
            case 'z':
                setResizableDim(atoi(optarg));
                break;
            case 'n':
                numIter = atoi(optarg);
                break;
            case 'r':
                if (strcmp(optarg, "gpu") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::GPU;
                }
                else if (strcmp(optarg, "dsp") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::DSP;
                }
                else if (strcmp(optarg, "cpu") == 0)
                {
                   runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
                }
                else if (strcmp(optarg, "cpu_ip8") == 0)
                {
                   runtime = zdl::DlSystem::Runtime_t::CPU_FIXED8_TF;
                }
                else if (strcmp(optarg, "gpu_fp16") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
                }
                else
                {
                   std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;

                }
                break;
            case 'c':
               usingInitCaching = true;
               break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(FAILURE);
        }
    }

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(FAILURE);
    }

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
        std::exit(FAILURE);
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
            std::exit(FAILURE);
#endif
            userBufferSourceType = GLBUFFER;
        }
        else
        {
            std::cout
                  << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
            std::exit(FAILURE);
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
       std::exit(FAILURE);
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;
#ifdef ANDROID
    CreateGLBuffer* glBuffer = nullptr;
    if (userBufferSourceType == GLBUFFER) {
        if(!checkGLCLInteropSupport()) {
            std::cerr << "Failed to get gl cl shared library" << std::endl;
            std::exit(1);
        }
        glBuffer = new CreateGLBuffer();
        glBuffer->setGPUPlatformConfig(platformConfig);
    }
#endif

    zdl::DlSystem::PerformanceProfile_t performanceProfile = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;

    snpe = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching, performanceProfile);
    if (snpe == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       std::exit(FAILURE);
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
        std::exit(FAILURE);
    }
    if (!logger->start()) {
        std::cerr << "Failed to start logger" << std::endl;
        std::exit(FAILURE);
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
            std::exit(1);
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
       std::cerr << "UserBuffer is not implemented yet" << std::endl;
       return FAILURE;
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
             std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
             if( batchSize > 1 )
                std::cout << "Batch " << i << ":" << std::endl;
             loadInputUserBufferTf8(applicationInputBuffers, snpe, inputs[i], inputMap);
             // Execute the input buffer map on the model with SNPE
             std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
             execStatus = snpe->execute(inputMap, outputMap);
             std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
             // Save the execution results only if successful
             if (execStatus == true)
             {
                saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, true);
             }
             else
             {
                std::cerr << "Error while executing the network." << std::endl;
             }

            std::cout << "Process elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin).count() << std::endl;
            std::cout << "Inference elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << std::endl;
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

                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBufferFloat(applicationInputBuffers, snpe, inputs[i]);
                // Execute the input buffer map on the model with SNPE
                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                execStatus = snpe->execute(inputMap, outputMap);
                std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
                // Save the execution results only if successful
                if (execStatus == true)
                {
                   saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, false);
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }

                std::cout << "Process elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin).count() << std::endl;
                std::cout << "Inference elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << std::endl;
             }
          }
#ifdef ANDROID
            if(userBufferSourceType  == GLBUFFER) {
                std::unordered_map<std::string, GLuint> applicationInputBuffers;
                createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe);
                GLuint glBuffers = 0;
                for(size_t i = 0; i < inputs.size(); i++) {
                    // Load input GL buffer(s) with values from file(s)
                    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                    glBuffers = glBuffer->convertImage2GLBuffer(inputs[i], bufSize);
                    loadInputUserBuffer(applicationInputBuffers, snpe, glBuffers);
                    // Execute the input buffer map on the model with SNPE
                    std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                    execStatus =  snpe->execute(inputMap, outputMap);
                    std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
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

                    std::cout << "Process elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin).count() << std::endl;
                    std::cout << "Inference elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << std::endl;
                }
            }
#endif
       }
    }
    else if(bufferType == ITENSOR)
    {
        // Create a log file
        const std::string logName = "latency.log";
        const std::string logPath = OutputDir + "/" + logName;

        std::ofstream logFile(logPath);
        if (!logFile.is_open()) {
            std::cerr << "Log file open fail: " << logName;
            return FAILURE;
        }
        else{
            std::cout << "Log file open: " << logPath;
            //logFile.setf(ios::fixed);
            logFile.precision(2);
        }

        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;

        for (size_t j = 0; j < numIter; j++) {
            for (size_t i = 0; i < inputs.size(); i++) {
                // Load input/output buffers with ITensor
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

                if(batchSize > 1)
                    std::cout << "Batch " << i << ":" << std::endl;
                std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[i]);
                // Execute the input tensor on the model with SNPE
                std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
                execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
                std::chrono::steady_clock::time_point end_= std::chrono::steady_clock::now();

                // Save the execution results if execution successful
                if (execStatus == true)
                {
                   //saveOutput(outputTensorMap, OutputDir, i * batchSize, batchSize);
                    std::cout << "Progress [" << j << "," << i << "] " << "Processing elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin).count() << std::endl;
                    logFile << j << "\t" << i << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin).count() << "\t" << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << std::endl;
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }
                std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
            }
        }
    }
    return SUCCESS;
}
