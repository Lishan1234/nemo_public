package android.example.testlibvpx;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {

    static{
        System.loadLibrary("jniVpxTest");
    }

    private native void vpxDecodeVideo(String videoSavedPath);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        vpxDecodeVideo("hello_world");
    }
}
