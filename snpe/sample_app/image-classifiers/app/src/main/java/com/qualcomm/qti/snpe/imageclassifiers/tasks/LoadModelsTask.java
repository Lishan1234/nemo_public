/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;

import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelCatalogueFragmentController;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileFilter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class LoadModelsTask extends AsyncTask<Void, Void, Set<Model>> {

    public static final String MODEL_DLC_FILE_NAME = "model.dlc";
    public static final String LR_IMAGES_FOLDER_NAME = "lr_images";
    public static final String HR_IMAGES_FOLDER_NAME = "hr_images";
    public static final String PNG_EXT = ".png";
    private static final String LOG_TAG = LoadModelsTask.class.getSimpleName();

    private final ModelCatalogueFragmentController mController;

    private final Context mContext;

    public LoadModelsTask(Context context, ModelCatalogueFragmentController controller) {
        mContext = context.getApplicationContext();
        mController = controller;
    }

    @Override
    protected Set<Model> doInBackground(Void... params) {
        final Set<Model> result = new LinkedHashSet<>();
        final File modelsRoot = mContext.getExternalFilesDir("models");
        if (modelsRoot != null) {
            result.addAll(createModels(modelsRoot));
        }
        return result;
    }

    @Override
    protected void onPostExecute(Set<Model> models) {
        mController.onModelsLoaded(models);
    }

    private Set<Model> createModels(File modelsRoot) {
        final Set<Model> models = new LinkedHashSet<>();
        final Set<String> availableModels = mController.getAvailableModels();
        for (File child : modelsRoot.listFiles()) {
            if (!child.isDirectory() || !availableModels.contains(child.getName())) {
                continue;
            }
            try {
                models.add(createModel(child));
            } catch (IOException e) {
                Log.e(LOG_TAG, "Failed to load model from model directory.", e);
            }
        }
        return models;
    }

    private Model createModel(File modelDir) throws IOException {
        final Model model = new Model();
        model.name = modelDir.getName();
        model.file = new File(modelDir, MODEL_DLC_FILE_NAME);
        final File Lrimages = new File(modelDir, LR_IMAGES_FOLDER_NAME);
        if (Lrimages.isDirectory()) {
            model.pngLrImages = Lrimages.listFiles(new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.getName().endsWith(PNG_EXT);
                }
            });
        }
        final File Hrimages = new File(modelDir, HR_IMAGES_FOLDER_NAME);
        if (Hrimages.isDirectory()) {
            model.pngHrImages = Hrimages.listFiles(new FileFilter() {
                @Override
                public boolean accept(File file) {
                    return file.getName().endsWith(PNG_EXT);
                }
            });
        }
        return model;
    }
}
