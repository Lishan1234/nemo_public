package android.example.testlibvpx;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class RawExtractor {
    private static final String LOG_TAG = RawExtractor.class.getSimpleName();
    //public static final String MODELS_ROOT_DIR = "nemo";
    private static final int CHUNK_SIZE = 1024;

    private RawExtractor() {}

    public static void execute(final Context context, final String rootDir, final String contentDir, final String videoDir, final int modelRawResId) {
        ZipInputStream zipInputStream = null;
        try {
            final File taskRoot = getOrCreateExternalDirectory(context, rootDir);
            final File contentRoot = createDirectory(taskRoot, contentDir);
            final File videoRoot = createDirectory(contentRoot, videoDir);
            if (modelExists(videoRoot)) {
                //TODO: remove saved files
                return;
            }

            zipInputStream = new ZipInputStream(context.getResources().openRawResource(modelRawResId));
            ZipEntry zipEntry;
            while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                final File entry = new File(videoRoot, zipEntry.getName());
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

    private static File getOrCreateExternalDirectory(Context context, final String rootDir) throws IOException {
        final File modelsRoot = context.getExternalFilesDir(rootDir);
        if (modelsRoot == null) {
            throw new IOException("Unable to access application external storage.");
        }

        if (!modelsRoot.isDirectory() && !modelsRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                    modelsRoot.getAbsolutePath());
        }
        return modelsRoot;
    }

    private static File createDirectory(File modelsRoot, String modelName) throws IOException {
        final File modelRoot = new File(modelsRoot, modelName);
        if (!modelRoot.isDirectory() && !modelRoot.mkdir()) {
            throw new IOException("Unable to create model root directory: " +
                    modelRoot.getAbsolutePath());
        }
        return modelRoot;
    }
}