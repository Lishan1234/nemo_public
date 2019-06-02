package com.example.batteryprofiler;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.source.ExtractorMediaSource;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.dash.DashChunkSource;
import com.google.android.exoplayer2.source.dash.DashMediaSource;
import com.google.android.exoplayer2.source.dash.DefaultDashChunkSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.upstream.DefaultHttpDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

import java.lang.reflect.InvocationTargetException;

public class MainActivity extends AppCompatActivity {

    Button button1;
    Button button2;
    Button button3;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @SuppressLint("PrivateApi")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button1 = findViewById(R.id.button1);
        button1.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), ExoPlayer.class);
                intent.putExtra("requestcode",1);
                startActivityForResult(intent, 1);
            }
        });

        button2 = findViewById(R.id.button2);
        button2.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), ExoPlayer.class);
                intent.putExtra("requestcode",2);
                startActivityForResult(intent, 2);
            }
        });

        button3 = findViewById(R.id.button3);
        button3.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getApplicationContext(), ExoPlayer.class);
                intent.putExtra("requestcode",3);
                startActivityForResult(intent, 3);
            }
        });
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data){
        if(requestCode == 1){
            if(resultCode == RESULT_OK) {
                Long start_battery = (Long) data.getLongExtra("start bat",0);
                Long end_battery = (Long) data.getLongExtra("end bat",0);
                int start_percent = data.getIntExtra("start percentage", 0);
                int end_percent = data.getIntExtra("end percentage", 0);

                EditText text = findViewById(R.id.text1);
                String result = "Start = " + start_battery +"(" +start_percent+"%)\n" + "End = " + end_battery + "(" +end_percent+"%)";
                text.setText(result);
            }
        }
        else if(requestCode == 2){
            if(resultCode == RESULT_OK) {
                Long start_battery = (Long) data.getLongExtra("start bat",0);
                Long end_battery = (Long) data.getLongExtra("end bat",0);
                int start_percent = data.getIntExtra("start percentage", 0);
                int end_percent = data.getIntExtra("end percentage", 0);

                EditText text = findViewById(R.id.text2);
                String result = "Start = " + start_battery +"(" +start_percent+"%)\n" + "End = " + end_battery + "(" +end_percent+"%)";
                text.setText(result);
            }
        }
        else if(requestCode == 3){
            if(resultCode == RESULT_OK) {
                Long start_battery = (Long) data.getLongExtra("start bat",0);
                Long end_battery = (Long) data.getLongExtra("end bat",0);
                int start_percent = data.getIntExtra("start percentage", 0);
                int end_percent = data.getIntExtra("end percentage", 0);

                EditText text = findViewById(R.id.text3);
                String result = "Start = " + start_battery +"(" +start_percent+"%)\n" + "End = " + end_battery + "(" +end_percent+"%)";
                text.setText(result);
            }
        }
    }


    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    public void test_code(){
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = getApplicationContext().registerReceiver(null,ifilter);

        int status = batteryStatus.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
        boolean isCharging = status==BatteryManager.BATTERY_STATUS_CHARGING ||
                status==BatteryManager.BATTERY_STATUS_FULL;

        int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL,-1);
        int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE,-1);

        Log.i("TAGG", level+"");
        Log.i("TAGG",scale+"");

        //
        BatteryManager mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);
        Long energy = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER);
        Long energy2 = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
        Log.i("TAGG",energy+"");
        Log.i("TAGG",energy2+"");
        Log.i("TAGG", "long.min-value = " + Long.MIN_VALUE);

        //
        BatteryManager bm = (BatteryManager) getSystemService(BATTERY_SERVICE);
        int batLevel = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        Log.e("TAGG",batLevel+"");


        //
        Object mPowerProfile_;
        String POWER_PROFILE_CLASS ="com.android.internal.os.PowerProfile";
        double batteryCapacity = -1;

        try {
            mPowerProfile_ = Class.forName(POWER_PROFILE_CLASS).getConstructor(Context.class).newInstance(this);
            batteryCapacity=(Double)Class.forName(POWER_PROFILE_CLASS).getMethod("getAveragePower", java.lang.String.class).invoke(mPowerProfile_, "battery.capacity");
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        Log.i("TAGG",batteryCapacity+"");

    }
}
