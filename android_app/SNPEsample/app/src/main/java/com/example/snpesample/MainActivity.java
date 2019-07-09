package com.example.snpesample;

import android.content.Intent;
import android.os.AsyncTask;
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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class MainActivity extends AppCompatActivity{

    Button lqx4_button;
    Button hqx4_button;

    @Override
    protected void onResume(){
        super.onResume();

        EditText edit = findViewById(R.id.time);
        edit.getText().clear();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        lqx4_button = findViewById(R.id.lqx4);
        lqx4_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String text = ((EditText)findViewById(R.id.time)).getText().toString();
                int minutes = Integer.parseInt(text);


                Intent intent = new Intent(getApplicationContext(),SNPEActivity.class);
                intent.putExtra("model",1);
                intent.putExtra("minutes",minutes);

                startActivity(intent);
            }
            }
        );

        hqx4_button = findViewById(R.id.hqx4);
        hqx4_button.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                String text = ((EditText)findViewById(R.id.time)).getText().toString();
                int minutes = Integer.parseInt(text);

                Intent intent = new Intent(getApplicationContext(),SNPEActivity.class);
                intent.putExtra("model",4);
                intent.putExtra("minutes",minutes);

                startActivity(intent);
            }
        });
    }


}