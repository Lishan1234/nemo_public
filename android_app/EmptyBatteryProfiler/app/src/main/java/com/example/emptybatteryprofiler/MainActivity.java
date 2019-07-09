package com.example.emptybatteryprofiler;

import android.content.Intent;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Handler;
import android.os.Message;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class MainActivity extends AppCompatActivity {

    BatteryManager mBatteryManager;

    long previous = 0;
    FileOutputStream fos = null;

    Button button;
    TextView textView;
    EditText editText;

    int minutes;
    int minutes_count;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


        //Create directories for log files if non-existent
        File logdir = new File("/sdcard","EmptyBattery");
        logdir.mkdirs();

        //initialise all
        mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);
        button = findViewById(R.id.button);
        textView = findViewById(R.id.text);
        editText = findViewById(R.id.duration);

        button.setOnClickListener(new Button.OnClickListener(){
            @Override
            public void onClick(View v) {
                minutes = Integer.parseInt(editText.getText().toString());
                minutes_count = minutes;

                //create log
                createLogFile(minutes);

                //create timers
                createTimers(minutes);

                //ui changes
                textView.setVisibility(TextView.VISIBLE);

                button.setVisibility(Button.INVISIBLE);
                button.setEnabled(false);

                editText.setVisibility(EditText.INVISIBLE);
                editText.setEnabled(false);
            }
        });
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

                Long battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
                int battery_percent = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);

                String entry = time + "," + Long.toString(battery) + "," + Integer.toString(battery_percent) + "%" + "," + Long.toString(previous-battery) + "\n";

                previous = battery;

                Log.e("TAG",entry);

                try {
                    fos.write(entry.getBytes());
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(minutes_count == 0){
                    //finish
                    //enable button
                    button.setVisibility(Button.VISIBLE);
                    button.setEnabled(true);

                    //enable text
                    editText.setVisibility(EditText.VISIBLE);
                    editText.setEnabled(true);

                    //enable textview
                    textView.setVisibility(TextView.INVISIBLE);

                    try {
                        fos.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
                minutes_count--;

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
    public void createLogFile(int mins){
        String fileName = "empty_battery" +"_"+ mins +"_mins";

        File file = new File("/sdcard/EmptyBattery/" + fileName + ".csv");

        try {
            fos = new FileOutputStream(file,false);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
