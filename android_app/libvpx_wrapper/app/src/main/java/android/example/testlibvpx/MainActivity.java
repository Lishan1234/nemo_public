package android.example.testlibvpx;

import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    public static native void vpxdec();
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

        //Execute a libvpx unit test
        vpxdec();
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
