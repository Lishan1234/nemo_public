//==============================================================================
//
//  Copyright (c) 2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstring>

#include "PreprocessInput.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

#include <android/log.h>

std::vector<std::vector<std::string>> preprocessInput(const char* filePath, size_t batchSize) {
    // Read lines from the input lists file
    // and store the paths to inputs in strings
    std::ifstream inputList(filePath);
    std::string fileLine;
    std::vector<std::string> lines;
    while (std::getline(inputList, fileLine)) {

        __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "line in input is : %s",fileLine.c_str());


        if (fileLine.empty()) continue;
        lines.push_back(fileLine);
    }

    __android_log_print(ANDROID_LOG_ERROR, "JNITAG", "batch size %zd",batchSize);
    // Store batches of inputs into vectors of strings
    std::vector<std::vector<std::string>> result;
    std::vector<std::string> batch;
    for(size_t i=0; i<lines.size(); i++) {
        if(batch.size()==batchSize) {
            result.push_back(batch);
            batch.clear();
        }
        batch.push_back(lines[i]);
    }
    result.push_back(batch);
    return result;
}
