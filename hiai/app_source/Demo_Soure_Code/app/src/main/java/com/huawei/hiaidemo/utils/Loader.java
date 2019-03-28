package com.huawei.hiaidemo.utils;

import java.io.File;
import java.io.FileFilter;

public class Loader {
    public static final String LR_IMAGES_FOLDER_NAME = "lr_images";
    public static final String HR_IMAGES_FOLDER_NAME = "hr_images";
    public static final String PNG_EXT = ".png";
    public static final String OFFLINE_EXT = ".cambricon";
    private static final String LOG_TAG = Loader.class.getSimpleName();

    public static Model createModel(final File modelDir, final String modelName) {
        final Model model = new Model();
        model.modelName = modelName;
        model.offlineModel = new File(modelDir, modelName + OFFLINE_EXT);
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