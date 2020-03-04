package com.example.exoplayernewlibvpx;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import static com.example.exoplayernewlibvpx.Constants.HOW_TO_PATH;
import static com.example.exoplayernewlibvpx.Constants.PRODUCT_REVIEW_PATH;
import static com.example.exoplayernewlibvpx.Constants.VLOGS_PATH;

public class ContentSelectionActivity extends AppCompatActivity {

    Spinner deviceSpinner;
    Spinner contentSpinner;
    Spinner modeSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_content_selection);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        requestPermissions();

        ArrayAdapter<CharSequence> deviceAdapter = ArrayAdapter.createFromResource(this, R.array.devices, android.R.layout.simple_spinner_item);
        deviceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        deviceSpinner = findViewById(R.id.select_device);
        deviceSpinner.setAdapter(deviceAdapter);

        ArrayAdapter<CharSequence> contentAdapter = ArrayAdapter.createFromResource(this, R.array.contents, android.R.layout.simple_spinner_item);
        contentAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        contentSpinner = findViewById(R.id.select_content);
        contentSpinner.setAdapter(contentAdapter);

        ArrayAdapter<CharSequence> modeAdapter = ArrayAdapter.createFromResource(this, R.array.modes, android.R.layout.simple_spinner_item);
        modeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner = findViewById(R.id.select_mode);
        modeSpinner.setAdapter(modeAdapter);

        findViewById(R.id.start).setOnClickListener((view)->{
                Intent intent = new Intent(ContentSelectionActivity.this, PlayerActivity.class);
                intent.putExtra("model_type",deviceSpinner.getSelectedItem().toString());
                intent.putExtra("content_type",contentSpinner.getSelectedItem().toString());
                intent.putExtra("mode_type",modeSpinner.getSelectedItem().toString());
                startActivity(intent);
            }
        );
    }

    private void requestPermissions(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},0);
        }
    }
}
