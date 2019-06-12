package com.example.snpesample;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class SNPEActivity  extends AppCompatActivity implements OnTaskCompleted{

    public native void function1(int model, int minutes);

    static{
        System.loadLibrary("snpeJNI");
    }

    Button start_button;
    Button back_button;
    TextView text;

    FileOutputStream fos;
    BatteryManager mBatteryManager;

    int minutes;
    int model;
    long previous = 0;

    private class LaunchMainFunction extends AsyncTask {
        private OnTaskCompleted listener;

        int minutes;
        int model;

        LaunchMainFunction(int model, int min, OnTaskCompleted listener){
            this.model = model;
            this.listener = listener;
            this.minutes = min;
        }

        @Override
        protected Object doInBackground(Object[] objects) {
            function1(this.model, this.minutes);
            return null;
        }

        @Override
        protected void onPostExecute(Object o){
            listener.onTaskCompleted();
        }
    }

    @Override
    public void onTaskCompleted() {
        //do something when async task completed
        //Change text to something
        text.setVisibility(TextView.VISIBLE);
        text.setText("DONE");

        back_button.setVisibility(Button.VISIBLE);
        back_button.setEnabled(true);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_snpe);

        //Create directories for log files if non-existent
        File logdir = new File("/sdcard","SNPEBattery");
        logdir.mkdirs();

        //battery init
        mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);

        //Get input from intent
        Intent intent = getIntent();
        model = intent.getIntExtra("model",0);
        minutes = intent.getIntExtra("minutes",0);

        //buttons init
        start_button = findViewById(R.id.start);
        back_button = findViewById(R.id.back);
        text = findViewById(R.id.progress);
        start_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //UI changes
                text.setVisibility(TextView.VISIBLE);
                start_button.setVisibility(Button.INVISIBLE);
                start_button.setEnabled(false);

                //create log file
                createLogFile(minutes);

                //create timers
                createTimers(minutes);

                //Launch native code in async
                new LaunchMainFunction(model, minutes, SNPEActivity.this).execute();
            }
        });
        back_button.setOnClickListener(new View.OnClickListener() {
           @Override
           public void onClick(View v) {
               finish();
           }
        });

    }

    //Make new log file and assign global stream
    public void createLogFile(int minutes){
        String fileName = "";

        switch(model){
            //lqx4
            case 1:
                fileName = "lq_dnn_x4";
                break;

            //lqx3
            case 2:
                fileName = "lq_dnn_x3";
                break;

            //lqx2
            case 3:
                fileName = "lq_dnn_x2";
                break;

            //hqx4
            case 4:
                fileName = "hq_dnn_x4";
                break;

            //hqx3
            case 5:
                fileName = "hq_dnn_x3";
                break;

            //hqx2
            case 6:
                fileName = "hq_dnn_x2";
                break;


            default:
                break;

        }

        fileName = "SNPE" + fileName + "_" + minutes + "_mins";

        File file = new File("/sdcard/SNPEBattery/" + fileName + ".csv");

        try {
            fos = new FileOutputStream(file,false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


    //Creates a timer for every minute that logs battery information
    public void createTimers(int minutes){
        //Create all the handler points
        Handler handler= new Handler(new Handler.Callback(){
            @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
            @Override
            public boolean handleMessage(Message msg) {
                //Record values
                Calendar cal = Calendar.getInstance();
                SimpleDateFormat dateFormat2 = new SimpleDateFormat("hh:mm:ss:SS");
                String time = dateFormat2.format(cal.getTime());

                long battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
                int battery_percent = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);

                String entry = time + "," + Long.toString(battery) + "," + Integer.toString(battery_percent) + "%" + "," + Long.toString(previous-battery) + "\n";

                previous = battery;

                Log.e("TAG",entry);

                try {
                    fos.write(entry.getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
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
}
