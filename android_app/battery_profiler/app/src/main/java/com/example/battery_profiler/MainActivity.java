package com.example.battery_profiler;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class MainActivity extends AppCompatActivity {

    EditText minutes_text;
    EditText fps_text;

    CheckBox exo_checkbox;
    CheckBox snpe_checkbox;

    RadioGroup exo_radio;
    RadioGroup snpe_radio;

    Button start_button;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Initialize
        fps_text = findViewById(R.id.fps);
        minutes_text = findViewById(R.id.minutes);
        exo_checkbox = findViewById(R.id.exo_check);
        snpe_checkbox = findViewById(R.id.snpe_check);
        exo_radio = findViewById(R.id.exo_radio);
        snpe_radio = findViewById(R.id.snpe_radio);
        start_button = findViewById(R.id.start);

        exo_checkbox.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if(exo_checkbox.isChecked()){
                    exo_radio.setVisibility(RadioGroup.VISIBLE);
                    exo_radio.setEnabled(true);
                }else{
                    exo_radio.setVisibility(RadioGroup.INVISIBLE);
                    exo_radio.setEnabled(false);
                }
            }
        });

        snpe_checkbox.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view) {
                if (snpe_checkbox.isChecked()) {
                    snpe_radio.setVisibility(RadioGroup.VISIBLE);
                    snpe_radio.setEnabled(true);
                    fps_text.setVisibility(EditText.VISIBLE);
                    fps_text.setEnabled(true);
                } else {
                    snpe_radio.setVisibility(RadioGroup.INVISIBLE);
                    snpe_radio.setEnabled(false);
                    fps_text.setVisibility(EditText.INVISIBLE);
                    fps_text.setEnabled(false);
                }
            }
        });

        start_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Intent intent = new Intent(getApplicationContext(),ExoSNPE.class);

                //Minutes
                if(minutes_text.getText().toString().matches("")){
                    Toast.makeText(getApplicationContext(), "Enter minutes", Toast.LENGTH_SHORT).show();
                    return;
                }
                int minutes = Integer.parseInt(minutes_text.getText().toString());
                intent.putExtra("minutes",minutes);

                //EXO
                if(exo_checkbox.isChecked()){
                    int exo_mode = 0;
                    int id = exo_radio.getCheckedRadioButtonId();
                    switch(id){
                        case R.id.exo_1:
                            exo_mode = 1;
                            break;
                        case R.id.exo_2:
                            exo_mode = 2;
                            break;
                        case R.id.exo_3:
                            exo_mode = 3;
                            break;
                        case R.id.exo_4:
                            exo_mode = 4;
                            break;
                        default:
                            //not selected, please select!
                            Toast toast=Toast.makeText(getApplicationContext(),"Pls select video quality", Toast.LENGTH_SHORT);
                            toast.show();
                            return;
                    }
                    intent.putExtra("exo_mode",exo_mode);
                }

                //SNPE
                if (snpe_checkbox.isChecked()) {
                    int snpe_mode = 0;
                    int id = snpe_radio.getCheckedRadioButtonId();
                    switch(id){
                        case R.id.snpe_lq:
                            snpe_mode = 1;
                            break;
                        case R.id.snpe_hq:
                            snpe_mode = 2;
                            break;
                        default:
                            //not selected, please select!
                            Toast toast=Toast.makeText(getApplicationContext(),"Pls select video quality", Toast.LENGTH_SHORT);
                            toast.show();
                            return;
                    }
                    intent.putExtra("snpe_mode",snpe_mode);

                    //FPS
                    if(fps_text.getText().toString().matches("")){
                        Toast.makeText(getApplicationContext(), "Enter fps limit", Toast.LENGTH_SHORT).show();
                        return;
                    }
                    int fps = Integer.parseInt(fps_text.getText().toString());
                    intent.putExtra("fps",fps);
                }
                Log.e("TAGG","start activity");
                startActivity(intent);
            }
        });
    }


}
