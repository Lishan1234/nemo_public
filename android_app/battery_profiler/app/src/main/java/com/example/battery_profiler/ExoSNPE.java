package com.example.battery_profiler;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.Player;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.source.ExtractorMediaSource;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.dash.DashChunkSource;
import com.google.android.exoplayer2.source.dash.DashMediaSource;
import com.google.android.exoplayer2.source.dash.DefaultDashChunkSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.AspectRatioFrameLayout;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.upstream.DefaultHttpDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class ExoSNPE extends AppCompatActivity{

    public native long function1(int model, int minutes, int fps);

    static{
        System.loadLibrary("snpeJNI");
    }

    private class LaunchMainFunction extends Thread{
        int minutes;
        int model;
        int fps;
        long count;

        LaunchMainFunction(int model, int min, int fps){
            Log.e("TAGG","name of thread ");
            this.model = model;
            this.minutes = min;
            this.fps = fps;
        }

        public void run(){
            count = function1(this.model, this.minutes,this.fps);

            String entry = count + "";

            Log.e("TAGG","Message: " + entry);

            try {
                fos.write(entry.getBytes());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    PlayerView mPlayerView;
    SimpleExoPlayer mSimpleExoPlayer;
    BatteryManager mBatteryManager;

    String video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm";

    int minutes;
    int fps;
    int exo_mode;
    int snpe_mode;

    long previous;
    FileOutputStream fos;

    LaunchMainFunction nativeFunctionThread;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Log.e("TAGG","create");

        //keep screen on
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //Create directories for log files if non-existent
        File logdir = new File("/sdcard","BatteryLogs");
        logdir.mkdirs();

        //Initialize variables
        previous = 0;
        mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);
        nativeFunctionThread = null;
        minutes = getIntent().getIntExtra("minutes",0);

        //Choose mode
        if(!getIntent().hasExtra("exo_mode")){
            fps = getIntent().getIntExtra("fps",0);
            snpe_mode = getIntent().getIntExtra("snpe_mode",0);

            String model = null;
            if(snpe_mode == 1){
                model = "lq";
            }else{
                model = "hq";
            }
            createLogFile("snpe_"+ model+"_"+minutes +"minutes");
            SNPE_only_mode();
        }
        else if(!getIntent().hasExtra("snpe_mode")){
            exo_mode = getIntent().getIntExtra("exo_mode",0);

            String quality = null;
            if(exo_mode == 1){
                quality = "360p";
            }
            else if(exo_mode == 2){
                quality = "720p";
            }
            else if(exo_mode == 3){
                quality = "1080p";
            }
            else if(exo_mode == 4){
                quality = "1080p";
            }
            createLogFile("exo_" + quality + "_" + minutes + "minutes");
            exo_only_mode();


        }else{
            fps = getIntent().getIntExtra("fps",0);
            snpe_mode = getIntent().getIntExtra("snpe_mode",0);
            exo_mode = getIntent().getIntExtra("exo_mode",0);


            String quality = null;
            if(exo_mode == 1){
                quality = "360p";
            }
            else if(exo_mode == 2){
                quality = "720p";
            }
            else if(exo_mode == 3){
                quality = "1080p";
            }
            else if(exo_mode == 4){
                quality = "1080p";
            }

            String model = null;
            if(snpe_mode == 1){
                model = "lq";
            }else{
                model = "hq";
            }
            createLogFile("snpe+exo_" + quality + "_" + model + "_" + minutes + "minutes");
            exo_and_SNPE_mode();

        }

        createTimers();
    }

    private void exo_only_mode(){
        setContentView(R.layout.exoplayer);

        //make landscape and hide navigation button
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

        //prepare player
        prep_exo();

        //play
        mSimpleExoPlayer.setRepeatMode(Player.REPEAT_MODE_ONE);
        mSimpleExoPlayer.setPlayWhenReady(true);
    }

    private void SNPE_only_mode(){
        setContentView(R.layout.snpe);

        //timer will probably call finish before SNPE breaks out of its time loop which may be problematic- use finish boolean.
        //timer can only finish if SNPE already done.
        //SNPE should also check if timer finished, and call finish by itself if timer done.

        nativeFunctionThread = new LaunchMainFunction(snpe_mode, minutes,fps);
        nativeFunctionThread.start();
    }

    private void exo_and_SNPE_mode(){
        setContentView(R.layout.exoplayer);

        //timer will probably call finish before SNPE breaks out of its time loop which may be problematic- use finish boolean.
        //timer can only finish if SNPE already done.
        //SNPE should also check if timer finished, and call finish by itself if timer done.

        nativeFunctionThread= new LaunchMainFunction(snpe_mode, minutes,fps);
        nativeFunctionThread.start();

        //make landscape and hide navigation button
//        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

        //prepare player
        prep_exo();

        //play
        mSimpleExoPlayer.setRepeatMode(Player.REPEAT_MODE_ONE);
        mSimpleExoPlayer.setPlayWhenReady(true);
    }


    public void prep_exo(){

        if(exo_mode == 1) {
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm";
        }
        else if(exo_mode==2){
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-720.webm";
        }
        else if(exo_mode==3){
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-1080.webm";
        }
        else if(exo_mode==4){
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-1080.webm";
        }

        //create player and set it to view
        mPlayerView = findViewById(R.id.video_view);

        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        mSimpleExoPlayer =
                ExoPlayerFactory.newSimpleInstance(this,
                        renderFactory,
                        new DefaultTrackSelector(),
                        new DefaultLoadControl(),
                        null);
        mPlayerView.setPlayer(mSimpleExoPlayer);

        //stretch to fit screen
        mPlayerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FILL);

        //Create  media source
        Uri uri = Uri.parse(video);

        //Create normal media source
        MediaSource mediaSource = createMediaSource(uri);

        //use media source
        mSimpleExoPlayer.prepare(mediaSource);
    }

    private MediaSource createMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"vp9testing"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }

    private DashMediaSource createDashMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultHttpDataSourceFactory("ExoPlayer");
        DashChunkSource.Factory dashChunkSourceFactory =
                new DefaultDashChunkSource.Factory(new DefaultHttpDataSourceFactory("ExoPlayer"));
        DashMediaSource mediaSource = new DashMediaSource.Factory(dashChunkSourceFactory, dataSourceFactory).createMediaSource(uri);
        return mediaSource;
    }

    //Creates a timer for every minute that logs battery information
    public void createTimers(){
        //Create all the handler points
        Handler handler= new Handler(new Handler.Callback(){
            @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
            @Override
            public boolean handleMessage(Message msg) {
                //write log entry
                Calendar cal = Calendar.getInstance();
                SimpleDateFormat dateFormat2 = new SimpleDateFormat("hh:mm:ss:SS");
                String time = dateFormat2.format(cal.getTime());

                Long battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
                int battery_percent = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);

                String entry = time + "," + Long.toString(battery) + "," + Integer.toString(battery_percent) + "%" + "," + Long.toString(previous-battery) + "\n";

                Log.e("TAGG","Message: " + entry);

                previous = battery;

                try {
                    fos.write(entry.getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
                }

                //check minutes
                if(minutes == 0){
                    try {
                        if(nativeFunctionThread != null)
                            nativeFunctionThread.join();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    finish();

                }else{
                    minutes--;
                }

                return false;
            }
        });

        Message message;
        for(int i = 0; i <= minutes;i++){
            message = handler.obtainMessage();
            handler.sendMessageDelayed(message, i*60000);
        }
    }

    //Make new log file and assign global stream
    public void createLogFile(String name){
        File file = new File("/sdcard/BatteryLogs/" + name + ".csv");

        try {
            fos = new FileOutputStream(file,false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}