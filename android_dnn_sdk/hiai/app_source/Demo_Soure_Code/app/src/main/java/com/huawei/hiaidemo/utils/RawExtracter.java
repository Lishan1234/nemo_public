package com.huawei.hiaidemo.utils;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class RawExtracter {
    private static final String LOG_TAG = RawExtracter.class.getSimpleName();
    public static final String MODELS_ROOT_DIR = "models";
    private static final int CHUNK_SIZE = 1024;

    private RawExtracter() {}

    public static void execute(final Context context, final String testName, final int modelRawResId) {
        ZipInputStream zipInputStream = null;
        try {
            final File modelsRoot = getOrCreateExternalModelsRootDirectory(context);
            final File modelRoot = createModelDirectory(modelsRoot, testName);
            if (modelExists(modelRoot)) {
                return;
            }

            zipInputStream = new ZipInputStream(context.getResources().openRawResource(modelRawResId));
            ZipEntry zipEntry;
            while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                final File entry = new File(modelRoot, zipEntry.getName());
                if (zipEntry.isDirectory()) {
                    doCreateDirectory(entry);
                } else {
                    doCreateFile(entry, zipInputStream);
                }
                zipInputStream.closeEntry();
            }
        } catch (IOException e) {
            Log.e(LOG_TAG, e.getMessage(), e);
            try {
                if (zipInputStream != null) {
                    zipInputStream.close();
                }
            } catch (IOException ignored) {}
        }
    }

    private static boolean modelExists(File modelRoot) {
        return modelRoot.listFiles().length > 0;
    }

    private static void doCreateFile(File file, ZipInputStream inputStream) throws IOException {
        final FileOutputStream outputStream = new FileOutputStream(file);
        final byte[] chunk = new byte[CHUNK_SIZE];
        int read;
        while ((read = inputStream.read(chunk)) != -1) {
            outputStream.write(chunk, 0, read);
        }
        outputStream.close();
    }

    private static void doCreateDirectory(File directory) throws IOException {
        if (!directory.mkdirs()) {
            throw new IOException("Can not create directory: " + directory.getAbsolutePath());
        }
    }

    private static File getOrCreateExternalModelsRootDirectory(Context context) throws IOException {
        final File modelsRoot = context.getExternalFilesDir(MODELS_ROOT_DIR);
        if (modelsRoot == null) {
            throw new IOException("Unable to access application external storage.");
        }

        if (!modelsRoot.isDirectory() && !modelsRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                    modelsRoot.getAbsolutePath());
        }
        return modelsRoot;
    }

    private static File createModelDirectory(File modelsRoot, String modelName) throws IOException {
        final File modelRoot = new File(modelsRoot, modelName);
        if (!modelRoot.isDirectory() && !modelRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                    modelRoot.getAbsolutePath());
        }
        return modelRoot;
    }
}
