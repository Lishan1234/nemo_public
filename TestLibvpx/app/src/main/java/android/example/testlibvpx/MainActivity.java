package android.example.testlibvpx;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {
    public native void vpxDecodeVideo(String videoSavedPath);

    static{
        System.loadLibrary("vpxtestJNI");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        vpxDecodeVideo("hello_world");
    }
}
