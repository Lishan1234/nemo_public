//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "CheckRuntime.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/DlVersion.hpp"
#include "DlSystem/DlEnums.hpp"
#include "DlSystem/String.hpp"

// Command line settings
zdl::DlSystem::Runtime_t checkRuntime(zdl::DlSystem::Runtime_t runtime)
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();

    std::cout << "SNPE Version: " << Version.asString().c_str() << std::endl; //Print Version number

    std::stringstream path;
    path << "/data/local/tmp/hyunho/lib;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp";
    if (setenv("ADSP_LIBRARY_PATH", path.str().c_str(), 1 /*override*/))
    {
        std::cerr << "setenv failed." << std::endl;
    }

    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime)) {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }

    return runtime;
}
bool checkGLCLInteropSupport()
{
    return zdl::SNPE::SNPEFactory::isGLCLInteropSupported();
}
