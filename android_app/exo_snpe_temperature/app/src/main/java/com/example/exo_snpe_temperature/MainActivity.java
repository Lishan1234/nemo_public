package com.example.exo_snpe_temperature;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.PermissionChecker;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.graphics.Matrix;
import android.graphics.drawable.Drawable;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbManager;
import android.media.MediaScannerConnection;
import android.media.Ringtone;
import android.media.RingtoneManager;
import android.net.Uri;
import android.opengl.GLSurfaceView;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.flir.flironesdk.Device;
import com.flir.flironesdk.FlirUsbDevice;
import com.flir.flironesdk.Frame;
import com.flir.flironesdk.FrameProcessor;
import com.flir.flironesdk.RenderedImage;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.EnumSet;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements Device.Delegate, Device.StreamDelegate, FrameProcessor.Delegate{

    GLSurfaceView thermalSurfaceView;
    private volatile Device flirOneDevice;
    private FrameProcessor frameProcessor;
    private Device.TuningState currentTuningState = Device.TuningState.Unknown;

    //added
    private static final long ONE_MINUTE = 60 * 1000;
    private static final long TEST_DURATION = ONE_MINUTE * 30;
    private static final long UPDATE_TEMP_FREQ_MS = 1000;
    private static final long UPDATE_PIC_FREQ_MS = ONE_MINUTE;

    private long startTime;
    private long lastUpdateTimeTemp;
    private long lastUpdateTimePic;
    private boolean testRunning = false;
    private String testPath;
    private TextView textView;
    private volatile FileOutputStream fos;
    private ArrayList<TemperaturePointer> temperaterPointerList;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        RenderedImage.ImageType defaultImageType = RenderedImage.ImageType.ThermalRGBA8888Image;
        frameProcessor = new FrameProcessor(this,this, EnumSet.of(RenderedImage.ImageType.ThermalRadiometricKelvinImage, RenderedImage.ImageType.ThermalRGBA8888Image),true);
        frameProcessor.setGLOutputMode(defaultImageType);
        frameProcessor.setImagePalette(RenderedImage.Palette.values()[2]);

        thermalSurfaceView = findViewById(R.id.imageView);
        thermalSurfaceView.setPreserveEGLContextOnPause(true);
        thermalSurfaceView.setEGLContextClientVersion(2);
        thermalSurfaceView.setRenderer(frameProcessor);
        thermalSurfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        thermalSurfaceView.setDebugFlags(GLSurfaceView.DEBUG_CHECK_GL_ERROR | GLSurfaceView.DEBUG_LOG_GL_CALLS);


        //permissions
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    0);
        }

        //make flirone directory
        File dir = new File("/sdcard/TemperatureTesting");
        if(!dir.exists()){
            dir.mkdir();
        }

        textView = findViewById(R.id.test_state);


        temperaterPointerList = new ArrayList<>();
        temperaterPointerList.add((TemperaturePointer)findViewById(R.id.temperature_pointer1));
        temperaterPointerList.add((TemperaturePointer)findViewById(R.id.temperature_pointer2));
        temperaterPointerList.add((TemperaturePointer)findViewById(R.id.temperature_pointer3));
        temperaterPointerList.add((TemperaturePointer)findViewById(R.id.temperature_pointer4));

        for(int i = 0; i < temperaterPointerList.size(); i++){
            TemperaturePointer tp = temperaterPointerList.get(i);
            tp.setId(i+1);
            FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT);
            params.gravity = Gravity.CENTER;
            tp.setLayoutParams(params);
            int imageInfo[] = getBitmapPositionInsideImageView(thermalSurfaceView);
            tp.setMovableBoundary(imageInfo[0],imageInfo[1],imageInfo[0] + imageInfo[2],imageInfo[1] + imageInfo[3], imageInfo[2], imageInfo[3]);
        }

    }

    @Override
    public void onStart(){
        super.onStart();
        Device.startDiscovery(this,this);
    }

    @Override
    public void onResume(){
        super.onResume();

        thermalSurfaceView.onResume();

        if(flirOneDevice != null){
            flirOneDevice.startFrameStream(this);
        }
    }


    @Override
    public void onPause(){
        super.onPause();

        thermalSurfaceView.onPause();
        if(flirOneDevice != null){
            flirOneDevice.stopFrameStream();
        }
    }

    @Override
    public void onStop(){
        super.onStop();
        Device.stopDiscovery();
        flirOneDevice = null;
    }

    @Override
    public void onDestroy(){
        super.onDestroy();

    }

    public void onBeginLog(View v) {

        v.setClickable(false);
        v.setVisibility(View.INVISIBLE);

        testPath = createLogFile();

        textView.setVisibility(View.VISIBLE);

        //start test for x minutes
        testRunning = true;
        startTime = System.currentTimeMillis();
        lastUpdateTimePic = startTime;
        lastUpdateTimeTemp = startTime;
    }

    private String createLogFile(){
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat dateFormat = new SimpleDateFormat("hh:mm:ss:SS");
        String time = dateFormat.format(cal.getTime());
        time = time.replaceAll(":","_");

        File dir = new File("/sdcard/TemperatureTesting/temperature_"+time);
        if(!dir.exists()){
            dir.mkdir();
        }

        File file = new File("/sdcard/TemperatureTesting/temperature_" + time + "/temperature.csv");
        try{
            fos = new FileOutputStream(file, false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return "/sdcard/TemperatureTesting/temperature_"+time;
    }

    public void onTuneClicked(View v){
        if(flirOneDevice != null){
            flirOneDevice.performTuning();
        }
    }


    /*** Device.Delegate callbacks ***/
    @Override
    public void onTuningStateChanged(Device.TuningState tuningState) {
        currentTuningState = tuningState;
    }

    @Override
    public void onAutomaticTuningChanged(boolean b) {
    }

    @Override
    public void onDeviceConnected(Device device) {
        flirOneDevice = device;
        flirOneDevice.startFrameStream(this);
    }

    @Override
    public void onDeviceDisconnected(Device device) {
        flirOneDevice = null;
    }


    /*** Device.StreamDelegate callbacks ***/
    @Override
    public void onFrameReceived(Frame frame) {

        if(currentTuningState != Device.TuningState.InProgress){
            frameProcessor.processFrame(frame, FrameProcessor.QueuingOption.CLEAR_QUEUED);
            thermalSurfaceView.requestRender();
        }
    }

    private boolean shouldUpdateTemp(){
        if(lastUpdateTimeTemp == startTime){
            lastUpdateTimeTemp++;
            return true;
        }

        return System.currentTimeMillis() - lastUpdateTimeTemp > UPDATE_TEMP_FREQ_MS;
    }

    private boolean shouldUpdatePic(){
        if(lastUpdateTimePic == startTime){
            lastUpdateTimePic++;
            return true;
        }

        return System.currentTimeMillis() - lastUpdateTimePic > UPDATE_PIC_FREQ_MS;
    }

    private boolean testFinished(){
        return System.currentTimeMillis() - startTime > TEST_DURATION;
    }

    /*** FrameProcessor.Delegate callbacks ***/
    @Override
    public void onFrameProcessed(final RenderedImage renderedImage) {
        double[] temps = new double[temperaterPointerList.size()];

        if(renderedImage.imageType() == RenderedImage.ImageType.ThermalRadiometricKelvinImage){
            for (int i = 0; i < temperaterPointerList.size(); i++) {
                final TemperaturePointer tp = temperaterPointerList.get(i);

                int[] thermalPixels = renderedImage.thermalPixelValues();
                int width = renderedImage.width();
                int height = renderedImage.height();

                float scaleX = ((float) width) / ((float) thermalSurfaceView.getWidth());
                float scaleY = ((float) height) / ((float) thermalSurfaceView.getHeight());

                int temp_x = (int) (tp.getWidthAdjusted() * scaleX);
                int temp_y = (int) (tp.getHeightAdjusted() * scaleY);
                final int centerPixelIndex = width * temp_y + temp_x;
                int[] centerPixelIndices = new int[]{
                        centerPixelIndex, centerPixelIndex - 1, centerPixelIndex + 1,
                        centerPixelIndex - width,
                        centerPixelIndex - width - 1,
                        centerPixelIndex - width + 1,
                        centerPixelIndex + width,
                        centerPixelIndex + width - 1,
                        centerPixelIndex + width + 1
                };
                double averageTemp = 0;
                for (int j = 0; j < centerPixelIndices.length; j++) {
                    int pixelValue = thermalPixels[centerPixelIndices[j]];
                    averageTemp += (((double) pixelValue) - averageTemp) / ((double) j + 1);
                }
                double averageC = (averageTemp / 100) - 273.15;
                NumberFormat numberFormat = NumberFormat.getInstance();
                numberFormat.setMaximumFractionDigits(2);
                numberFormat.setMinimumFractionDigits(2);
                final String spotMeterValue = numberFormat.format(averageC) + "ºC";
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        tp.setTemperatureText(spotMeterValue);
                    }
                });
                temps[i] = averageC;
            }
        }



        //if test running and update frequency satisfied
        if (testRunning) {
            if (shouldUpdateTemp() && renderedImage.imageType() == RenderedImage.ImageType.ThermalRadiometricKelvinImage) {
                Calendar cal = Calendar.getInstance();
                SimpleDateFormat dateFormat = new SimpleDateFormat("hh:mm:ss:SS");
                String time = dateFormat.format(cal.getTime());
                NumberFormat numberFormat = NumberFormat.getInstance();
                String entry = "";
                for (int i = 0; i < temps.length; i++) {
                    entry += time + "," + (i + 1) + "," + numberFormat.format(temps[i]) + "\n";
                }
                try {
                    fos.write(entry.getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                lastUpdateTimeTemp = System.currentTimeMillis();
            }

            if (shouldUpdatePic() && renderedImage.imageType() == RenderedImage.ImageType.ThermalRGBA8888Image) {
                final Context context = this;
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ssZ", Locale.getDefault());
                        String formatedDate = sdf.format(new Date());
                        String fileName = "FLIROne-" + formatedDate + ".jpg";
                        try {
                            String path = testPath + "/" + fileName;
                            renderedImage.getFrame().save(new File(path), frameProcessor);

                            //check if saved ok
                            MediaScannerConnection.scanFile(context,
                                    new String[]{path}, null,
                                    new MediaScannerConnection.OnScanCompletedListener() {
                                        @Override
                                        public void onScanCompleted(String s, Uri uri) {
                                            Log.i("ExternalStorage", "Scanned " + s + ":");
                                            Log.i("ExternalStorage", "-> uri=" + uri);
                                        }
                                    });
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }).start();
                lastUpdateTimePic = System.currentTimeMillis();
            }
        }

        //finish
        if (testRunning && testFinished()) {
            testRunning = false;
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    //finish
                    textView.setText("Test finished");
                    try {
                        fos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });
            playSound();
        }
    }

    /*
     * 0: left, 1: top, 2: width, 3: height
     */
    public static int[] getBitmapPositionInsideImageView(GLSurfaceView surfaceView) {
        int[] ret = new int[4];
        ret[2] = surfaceView.getWidth();
        ret[3] = surfaceView.getHeight();
        ret[0] = (int) surfaceView.getX();
        ret[1] = (int) surfaceView.getY();
        return ret;
    }

    public void playSound(){
        try {
            Uri notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);
            Ringtone r = RingtoneManager.getRingtone(getApplicationContext(), notification);
            r.play();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
