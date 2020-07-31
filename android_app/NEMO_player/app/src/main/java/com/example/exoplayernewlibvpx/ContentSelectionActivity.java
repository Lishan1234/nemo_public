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

public class ContentSelectionActivity extends AppCompatActivity {

    Spinner qualitySpinner;
    Spinner resolutionSpinner;
    Spinner modeSpinner;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_content_selection);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        requestPermissions();

        ArrayAdapter<CharSequence> deviceAdapter = ArrayAdapter.createFromResource(this, R.array.quality, android.R.layout.simple_spinner_item);
        deviceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        qualitySpinner = findViewById(R.id.select_device);
        qualitySpinner.setAdapter(deviceAdapter);

        ArrayAdapter<CharSequence> contentAdapter = ArrayAdapter.createFromResource(this, R.array.resolution, android.R.layout.simple_spinner_item);
        contentAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        resolutionSpinner = findViewById(R.id.select_content);
        resolutionSpinner.setAdapter(contentAdapter);

        ArrayAdapter<CharSequence> modeAdapter = ArrayAdapter.createFromResource(this, R.array.mode, android.R.layout.simple_spinner_item);
        modeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner = findViewById(R.id.select_mode);
        modeSpinner.setAdapter(modeAdapter);

        findViewById(R.id.start).setOnClickListener((view)->{
                Intent intent = new Intent(ContentSelectionActivity.this, PlayerActivity.class);
                intent.putExtra("quality", qualitySpinner.getSelectedItem().toString());
                intent.putExtra("resolution", resolutionSpinner.getSelectedItem().toString());
                intent.putExtra("mode",modeSpinner.getSelectedItem().toString());
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
