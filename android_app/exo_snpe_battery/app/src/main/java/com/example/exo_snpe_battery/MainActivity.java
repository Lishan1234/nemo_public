package com.example.exo_snpe_battery;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.WindowManager;

import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.Player;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.ProgressiveMediaSource;
import com.google.android.exoplayer2.ui.AspectRatioFrameLayout;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.Queue;

import static com.example.exo_snpe_battery.Constants.COOLDOWN_DURATION_MS;
import static com.example.exo_snpe_battery.Constants.MESSAGE_KEEP_CHARGING;
import static com.example.exo_snpe_battery.Constants.MESSAGE_LOG_BATTERY;
import static com.example.exo_snpe_battery.Constants.MESSAGE_NEXT_REPEAT;
import static com.example.exo_snpe_battery.Constants.MESSAGE_NEXT_TEST;
import static com.example.exo_snpe_battery.Constants.MESSAGE_NO_BAT;
import static com.example.exo_snpe_battery.Constants.ONE_MINUTE_MS;
import static com.example.exo_snpe_battery.Constants.START_THRESHOLD;
import static com.example.exo_snpe_battery.Constants.TestType.HIGH;
import static com.example.exo_snpe_battery.Constants.TestType.LOW;
import static com.example.exo_snpe_battery.Constants.TestType.MEDIUM;
import static com.example.exo_snpe_battery.Constants.TestType.NONE;

public class MainActivity extends AppCompatActivity {

    public native long jniFunction(int minutes, String dlc, String input, String output, String log, boolean doLog);

    static{
        System.loadLibrary("snpeJNI");
    }

    private class TimerHandler extends Handler{

        @Override
        public void handleMessage(@NonNull Message msg) {
            switch(msg.what){
                case MESSAGE_NO_BAT:
                    //clean up snpe
                    if(msg.getData().getSerializable("test_type") != NONE){
                        //wait for snpe thread
                        try {
                            if(mThread != null) {
                                mThread.join();
                            }
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    mSimpleExoPlayer.stop();


                case MESSAGE_KEEP_CHARGING:

                    Log.e("TAG","Message 1");
                    Long battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
                    if(battery >= START_THRESHOLD){
                        //launch test
                        Message message = handler.obtainMessage();
                        message.copyFrom(msg);
                        disableCharging();
                        beginSingleTest(message);
                        Log.e("TAG","begin test");

                    }else{
                        //keep charging
                        Message message = handler.obtainMessage();
                        message.copyFrom(msg);
                        this.sendMessageDelayed(message, ONE_MINUTE_MS);
                        Log.e("TAG","Keep charging to 100%, currently at = " + battery);
                    }
                    break;
                case MESSAGE_LOG_BATTERY:

                    Log.e("TAG","Message 2");

                    //Log battery and update previous battery
                    long currentBattery = logBattery(msg.getData().getLong("previous_battery"));

                    int minutes = msg.getData().getInt("minutes");
                    if(minutes == 0){

                        Log.e("TAG","test done");

                        //clean up snpe
                        if(msg.getData().getSerializable("test_type") != NONE){
                            //wait for snpe thread
                            try {
                                if(mThread != null) {
                                    mThread.join();
                                }
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        }

                        //clean up exo
                        mSimpleExoPlayer.stop();

                        //clean up file
                        try {
                            fos.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        //check if all repetitions are done
                        int repeat = msg.getData().getInt("repeat");
                        if(repeat == 1){
                            if(!workQueue.isEmpty()){
                                Log.e("TAG","next test unit");
                                //todo - sleep for cooldown duration
                                Message message = this.obtainMessage();
                                message.what = MESSAGE_NEXT_TEST;
                                this.sendMessageDelayed(message, COOLDOWN_DURATION_MS);
//                                continueTests();
                            }
                        }else{
                            Log.e("TAG","repeat");

                            //todo - sleep for cooldown duration
                            Message message = this.obtainMessage();
                            message.copyFrom(msg);
                            message.what = MESSAGE_NEXT_REPEAT;
                            message.getData().putInt("repeat",repeat-1);
                            this.sendMessageDelayed(message, COOLDOWN_DURATION_MS);
//                            launchSingleTest((Constants.TestType) msg.getData().getSerializable("test_type"),msg.getData().getInt("original_minutes"),repeat - 1);
                        }


                    }else{
                        Log.e("TAG","continue logging");

                        Message message = this.obtainMessage();
                        message.copyFrom(msg);

                        //update minutes and previous battery
                        message.getData().putInt("minutes",minutes-1);
                        message.getData().putLong("previous_battery", currentBattery);

                        this.sendMessageDelayed(message, ONE_MINUTE_MS);
                    }
                    break;
                case MESSAGE_NEXT_TEST:
                    continueTests();
                    break;
                case MESSAGE_NEXT_REPEAT:
                    launchSingleTest((Constants.TestType)msg.getData().getSerializable("test_type"),msg.getData().getInt("original_minutes"),msg.getData().getInt("repeat"));
                    break;
            }
        }
    }

    //set up jni stuff
    private class SNPEThread extends Thread{
        private int minutes;
        private String dlc;
        private String input;
        private String output;
        private String log;
        private boolean doLog;

        public SNPEThread(int minutes, String dlc, String input, String output, String log, boolean doLog){
            this.minutes = minutes;
            this.dlc = dlc;
            this.input = input;
            this.output = output;
            this.log = log;
            this.doLog = doLog;
        }

        @Override
        public void run(){
            jniFunction(minutes, dlc, input,output,log, doLog);
        }
    }

    PlayerView mPlayerView;
    SimpleExoPlayer mSimpleExoPlayer;
    BatteryManager mBatteryManager;
    SNPEThread mThread;
    FileOutputStream fos;
    TimerHandler handler;
    Queue<BatteryTestUnit> workQueue;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.e("TAG", "onCreate");

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        //init stuff
        mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);
        handler = new TimerHandler();
        prepareExoPlayer();

        //add tests
        //NO_BAT mode - you should pre charge the phones before the tests
        //Other modes - auto charge so it doesn't matter
        workQueue = new LinkedList<>();
//        workQueue.add(new BatteryTestUnit(LOW, 3, 10,true));
//        workQueue.add(new BatteryTestUnit(MEDIUM,3,10,true));
//        workQueue.add(new BatteryTestUnit(HIGH,3,10,true));
//        workQueue.add(new BatteryTestUnit(NONE,3,10,true));
        workQueue.add(new BatteryTestUnit(NONE,1,31,false));

        //begin
        continueTests();
    }

    private void continueTests(){
        if(!workQueue.isEmpty()){
            BatteryTestUnit unit = workQueue.remove();
            if(!unit.log){
                launchNoBatTest(unit.testType, unit.minutes);
            }else{
                launchSingleTest(unit.testType, unit.minutes, unit.repeat);
            }
        }
    }

    private void launchNoBatTest(Constants.TestType testType, int minutes){
        //no logging required. just simply run exo and snpe for temperature testing

        switch(testType){
            case LOW:
                mThread = new SNPEThread(minutes, Constants.lowDlc, Constants.input, Constants.output, null, false);
                mThread.start();

                break;
            case MEDIUM:
                mThread = new SNPEThread(minutes, Constants.mediumDlc, Constants.input, Constants.output, null, false);
                mThread.start();
                break;
            case HIGH:
                mThread = new SNPEThread(minutes, Constants.highDlc, Constants.input, Constants.output, null, false);
                mThread.start();
                break;
            case NONE:
                break;
        }

        mSimpleExoPlayer.retry();
        mSimpleExoPlayer.setPlayWhenReady(true);

        //
        Message message = handler.obtainMessage();
        message.what = MESSAGE_NO_BAT;
        Bundle bundle = new Bundle();
        bundle.putSerializable("test_type",testType);
        message.setData(bundle);
        handler.sendMessageDelayed(message, minutes * ONE_MINUTE_MS);
    }

    private void launchSingleTest(Constants.TestType testType, int minutes, int repeat){
        //charge to 100 and begin
        enableCharging();

        Message message = handler.obtainMessage();
        Bundle bundle = new Bundle();
        bundle.putSerializable("test_type", testType);
        bundle.putInt("minutes",minutes);
        bundle.putInt("repeat", repeat);

        bundle.putInt("original_minutes",minutes);

        message.setData(bundle);
        message.what = MESSAGE_KEEP_CHARGING;
        handler.sendMessage(message);
    }

    private void beginSingleTest(Message msg){

        String dir;

        switch( (Constants.TestType) msg.getData().getSerializable("test_type")){
            case LOW:

                dir = createLogFile("low");
                //start snpe and exoplayer
                mThread = new SNPEThread(msg.getData().getInt("minutes"), Constants.lowDlc, Constants.input, Constants.output, dir, true);
                mThread.start();

                mSimpleExoPlayer.retry();
                mSimpleExoPlayer.setPlayWhenReady(true);

                msg.what = MESSAGE_LOG_BATTERY;
                msg.getData().putLong("previous_battery",0);
                handler.sendMessage(msg);

                break;
            case MEDIUM:

                dir = createLogFile("medium");
                //start snpe and exoplayer
                mThread = new SNPEThread(msg.getData().getInt("minutes"), Constants.mediumDlc, Constants.input, Constants.output, dir, true);
                mThread.start();

                mSimpleExoPlayer.retry();
                mSimpleExoPlayer.setPlayWhenReady(true);

                msg.what = MESSAGE_LOG_BATTERY;
                msg.getData().putLong("previous_battery",0);
                handler.sendMessage(msg);
                break;
            case HIGH:
                dir = createLogFile("high");
                //start snpe and exoplayer
                mThread = new SNPEThread(msg.getData().getInt("minutes"), Constants.highDlc, Constants.input, Constants.output, dir, true);
                mThread.start();

                mSimpleExoPlayer.retry();
                mSimpleExoPlayer.setPlayWhenReady(true);

                msg.what = MESSAGE_LOG_BATTERY;
                msg.getData().putLong("previous_battery",0);
                handler.sendMessage(msg);
                break;
            case NONE:
                break;
            default:
                break;
        }
    }


    //return directory name
    private String createLogFile(String testName){
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat dateFormat = new SimpleDateFormat("hh:mm:ss:SS");
        String time = dateFormat.format(cal.getTime());
        time = time.replaceAll(":","_");

        File dir = new File("/sdcard/BatteryTesting/log/"+testName + time);

        if(!dir.exists()){
                dir.mkdir();
        }

        File file = new File("/sdcard/BatteryTesting/log/" + testName + time + "/battery.csv");
        try {
            fos = new FileOutputStream(file, false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return "sdcard/BatteryTesting/log/"+testName + time;
    }

    private void prepareExoPlayer(){
        mPlayerView = findViewById(R.id.video_view);
        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        mSimpleExoPlayer = new SimpleExoPlayer.Builder(this, renderFactory).build();
        mSimpleExoPlayer.setRepeatMode(Player.REPEAT_MODE_ALL);
        mPlayerView.setPlayer(mSimpleExoPlayer);
        mPlayerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FIXED_HEIGHT);
        mSimpleExoPlayer.prepare(createMediaSource());
    }

    private MediaSource createMediaSource(){
        File file = new File(Constants.videoPath);
        Uri uri = Uri.fromFile(file);
        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this, "vp9testing"));
        MediaSource videoSource = new ProgressiveMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }


    private long logBattery(long previous){
        //write log entry
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat dateFormat = new SimpleDateFormat("hh:mm:ss:SS");
        String time = dateFormat.format(cal.getTime());

        Long battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
        int batteryPercent = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);

        String entry = time + "," + battery +"," + batteryPercent + "%" + "," + Long.toString(previous-battery) + "\n";

        Log.e("BAT",entry);

        try{
            fos.write(entry.getBytes());
        }catch(IOException e){
            e.printStackTrace();
        }

        return battery;
    }


    private void enableCharging(){

        try {
            Process process = Runtime.getRuntime().exec("su");
            DataOutputStream outputStream = new DataOutputStream(process.getOutputStream());
            outputStream.writeBytes("echo 1 > /sys/class/power_supply/battery/charging_enabled");
            outputStream.flush();
            outputStream.close();
            process.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.e("TAG","enabled charging");
    }

    private void disableCharging(){
        try {
            Process process = Runtime.getRuntime().exec("su");
            DataOutputStream outputStream = new DataOutputStream(process.getOutputStream());
            outputStream.writeBytes("echo 0 > /sys/class/power_supply/battery/charging_enabled");
            outputStream.flush();
            outputStream.close();
            process.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Log.e("TAG","disabled charging");
    }


}
