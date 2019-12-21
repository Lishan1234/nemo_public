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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Execute a libvpx unit test
        vpxdec();
        Log.i(TAG, "MainActivity ends");
    }
}
