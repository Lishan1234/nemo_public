package android.example.testlibvpx;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import java.io.File;

public class MainActivity extends AppCompatActivity {
    public native void vpxDecodeVideo(String videoSavedPath, String logPath);
    private static final String TAG = MainActivity.class.getSimpleName();

    static{
        System.loadLibrary("vpxtestJNI");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Unzip and copy data
        int resourceId = this.getResources().getIdentifier("data", "raw", this.getPackageName());
        RawExtractor.execute(this, "data", resourceId);

        //Get video path
        File videoDir = getExternalFilesDir("mobinas" + File.separator + "data");
        File logDir = getExternalFilesDir("mobinas" + File.separator + "log");
        String videoPath = videoDir.getAbsolutePath() + File.separator + "test.webm";
        String logPath = logDir.getAbsolutePath();

        //Execute a libvpx unit test
        vpxDecodeVideo(videoPath, logPath);

        Log.i(TAG, "MainActivity ends");
    }
}
