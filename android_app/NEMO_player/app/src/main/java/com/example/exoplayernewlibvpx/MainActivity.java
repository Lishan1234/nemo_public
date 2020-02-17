package com.example.exoplayernewlibvpx;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.Player;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.source.ExtractorMediaSource;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.AspectRatioFrameLayout;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

        requestPermissions();

        setupDirectories();

        exoplayer();
    }


    private void exoplayer(){
        setContentView(R.layout.exoplayer);

        PlayerView playerView = findViewById(R.id.player);

        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        SimpleExoPlayer simpleExoPlayer =
                ExoPlayerFactory.newSimpleInstance(this,
                        renderFactory,
                        new DefaultTrackSelector(),
                        new DefaultLoadControl(),
                        null);
        playerView.setPlayer(simpleExoPlayer);

        playerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FIXED_HEIGHT);

        MediaSource mediaSource = createLocalMediaSource();

        simpleExoPlayer.prepare(mediaSource);
        simpleExoPlayer.setPlayWhenReady(true);

        simpleExoPlayer.addListener(new Player.EventListener() {
            @Override
            public void onPlayerStateChanged(boolean playWhenReady, int playbackState) {
                if(playbackState == Player.STATE_ENDED){
                    simpleExoPlayer.stop(true);
                }
            }
        });
    }

    private MediaSource createLocalMediaSource(){
        String video = "/storage/emulated/0/Android/data/android.example.testlibvpx/files/video/240p_s0_d60_encoded.webm";
        File file = new File(video);
        Uri uri = Uri.fromFile(file);

        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"ExoPlayer"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }


    private void requestPermissions(){
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this,new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},0);
        }
    }

    private void setupDirectories(){
        File dataDir = new File("/storage/emulated/0/Android/data","android.example.testlibvpx");
        if(!dataDir.exists()){
            dataDir.mkdir();

            File fileDir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx","files");
            if(!fileDir.exists()){
                fileDir.mkdir();

                //Make inner directory structures
                File videoDir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files","video");
                File checkpointDir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files","checkpoint");
                File imageDir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files","image");
                File logDir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files","log");
                videoDir.mkdir();
                checkpointDir.mkdir();
                imageDir.mkdir();
                logDir.mkdir();

                //Add model and video from android resources
                File edsr64Dir = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files/checkpoint","EDSR_S_B8_F64_S4");
                edsr64Dir.mkdir();
                File model = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files/checkpoint/EDSR_S_B8_F64_S4","ckpt-100.dlc");

                File video = new File("/storage/emulated/0/Android/data/android.example.testlibvpx/files/video","240p_s0_d60_encoded.webm");

                try {
                    InputStream modelInputStream = getResources().openRawResource(R.raw.ckpt_100);
                    OutputStream modelOutputStream = new FileOutputStream(model);
                    byte[] data = new byte[modelInputStream.available()];
                    modelInputStream.read(data);
                    modelOutputStream.write(data);
                    modelInputStream.close();
                    modelOutputStream.close();

                    InputStream videoInputStream = getResources().openRawResource(R.raw.video);
                    OutputStream videoOutputStream = new FileOutputStream(video);
                    byte[] videoData = new byte[videoInputStream.available()];
                    videoInputStream.read(videoData);
                    videoOutputStream.write(videoData);
                    videoInputStream.close();
                    videoOutputStream.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    //This version should be used when libvpx is changed to receive file paths from java side.
    private void setupDirectoriesSafe(){

        File fileDir = new File(this.getExternalFilesDir(null),"files");
        //Files have not been initialized yet.
        if(!fileDir.exists()){
            fileDir.mkdir();

            //Make all directory structures
            File videoDir = new File(this.getExternalFilesDir("files"),"video");
            File checkpointDir = new File(this.getExternalFilesDir("files"),"checkpoint");
            File imageDir = new File(this.getExternalFilesDir("files"),"image");
            File logDir = new File(this.getExternalFilesDir("files"),"log");
            videoDir.mkdir();
            checkpointDir.mkdir();
            imageDir.mkdir();
            logDir.mkdir();

            //Add model from android resources
            InputStream inputStream = getResources().openRawResource(R.raw.ckpt_100);

        }


    }
}