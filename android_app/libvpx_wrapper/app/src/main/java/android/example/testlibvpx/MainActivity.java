package android.example.testlibvpx;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    public static native void vpxDecodeVideo(String videoSavedPath, String logPath, String framePath, String serializePath, int target_resolution, int scale);

    private static final String TAG = MainActivity.class.getSimpleName();

    static{
        System.loadLibrary("vpxtestJNI");
    }

    static void cleanDirectory(File directory) {
        File[] files = directory.listFiles();
        for (File file: files)
        {
            if (!file.delete())
            {
                System.out.println("Failed to delete "+ file);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final String name = "mobinas";
        final String content = "movie";
        final int target_resolution = 1080;
        final int scale = 4;

//        Unzip and copy data
//        int resourceId = this.getResources().getIdentifier("starcraft", "raw", this.getPackageName());
//        RawExtractor.execute(this, name, content, "video", resourceId); //TODO: use this as a content name

        //Get video path
        File videoDir = getExternalFilesDir(name + File.separator + content + File.separator + "video");
        File logDir = getExternalFilesDir(name + File.separator + content + File.separator + "log");
        File frameDir = getExternalFilesDir(name + File.separator + content + File.separator + "frame");
        File serializeDir = getExternalFilesDir(name + File.separator + content + File.separator + "serialize");

        try{
            if(!logDir.exists() && logDir.isDirectory()) doCreateDirectory(logDir);
            if(frameDir.exists() && logDir.isDirectory()) cleanDirectory(frameDir); //remove previous results
            if(!frameDir.exists() && frameDir.isDirectory()) doCreateDirectory(frameDir);
            if(!serializeDir.exists() && serializeDir.isDirectory()) doCreateDirectory(serializeDir);
        }
        catch (IOException e)
        {
            Log.e(TAG, e.getMessage(), e);
        }

        String videoPath = videoDir.getAbsolutePath();
        String logPath = logDir.getAbsolutePath();
        String framePath = frameDir.getAbsolutePath();
        String serializePath = serializeDir.getAbsolutePath();

        //Execute a libvpx unit test
        vpxDecodeVideo(videoPath, logPath, framePath, serializePath, target_resolution, scale);
        //helloworld();

        Log.i(TAG, "MainActivity ends");
    }

    private static void doCreateDirectory(File directory) throws IOException {
        if (!directory.mkdirs()) {
            throw new IOException("Can not create directory: " + directory.getAbsolutePath());
        }
    }

    private static boolean modelExists(File modelRoot) {
        return modelRoot.listFiles().length > 0;
    }
}
