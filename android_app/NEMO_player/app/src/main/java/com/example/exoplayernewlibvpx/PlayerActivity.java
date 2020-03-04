package com.example.exoplayernewlibvpx;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;

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

import static com.example.exoplayernewlibvpx.Constants.HOW_TO_PATH;
import static com.example.exoplayernewlibvpx.Constants.MESSAGE_EXO_STOP;
import static com.example.exoplayernewlibvpx.Constants.ONE_MINUTE_MS;
import static com.example.exoplayernewlibvpx.Constants.PRODUCT_REVIEW_PATH;
import static com.example.exoplayernewlibvpx.Constants.UNBOXING_PATH;
import static com.example.exoplayernewlibvpx.Constants.VIDEO_RELATIVE_PATH;
import static com.example.exoplayernewlibvpx.Constants.VLOGS_PATH;


public class PlayerActivity extends AppCompatActivity {

    SimpleExoPlayer mSimpleExoPlayer;
    ExoPlayerHandler mExoPlayerHandler;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_player);

        setupExoPlayer(getIntent().getStringExtra("model_type"),
                getIntent().getStringExtra("content_type"),
                getIntent().getStringExtra("mode_type"));

        loopExoPlayer(7);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if(mSimpleExoPlayer!=null){
            mSimpleExoPlayer.stop(true);
            mSimpleExoPlayer.release();
        }
    }

    private void setupExoPlayer(String model, String content, String mode){

        String contentPath = "";
        if(content.equals("product_review")){
            contentPath = PRODUCT_REVIEW_PATH;
        } else if (content.equals("vlogs")) {
            contentPath = VLOGS_PATH;
        } else if (content.equals("how_to")) {
            contentPath = HOW_TO_PATH;
        } else if(content.equals("unboxing")){
            contentPath = UNBOXING_PATH;
        }

        int decodeMode = 0;
        if(mode.equals("Decode")){
            decodeMode = 0;
        } else if (mode.equals("Decode-SR")){
            decodeMode = 1;
        } else if (mode.equals("Decode-Cache")){
            decodeMode = 2;
        }

        PlayerView playerView = findViewById(R.id.player);

        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this,contentPath,model,decodeMode);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        mSimpleExoPlayer =
                ExoPlayerFactory.newSimpleInstance(this,
                        renderFactory,
                        new DefaultTrackSelector(),
                        new DefaultLoadControl(),
                        null);
        playerView.setPlayer(mSimpleExoPlayer);

        playerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FIXED_HEIGHT);

        MediaSource mediaSource = createLocalMediaSource(contentPath);

        mSimpleExoPlayer.prepare(mediaSource);
        mSimpleExoPlayer.setPlayWhenReady(true);

        mSimpleExoPlayer.addListener(new Player.EventListener() {
            @Override
            public void onPlayerStateChanged(boolean playWhenReady, int playbackState) {
                if(playbackState == Player.STATE_ENDED){
                    mSimpleExoPlayer.stop(true);
                }
            }
        });
    }

    private void loopExoPlayer(int minutes){
        mSimpleExoPlayer.setRepeatMode(Player.REPEAT_MODE_ALL);
        mExoPlayerHandler = new ExoPlayerHandler();
        Message message = mExoPlayerHandler.obtainMessage();
        message.what = MESSAGE_EXO_STOP;
        mExoPlayerHandler.sendMessageDelayed(message,minutes*ONE_MINUTE_MS);
    }

    private MediaSource createLocalMediaSource(String contentPath){
        File file = new File(contentPath + VIDEO_RELATIVE_PATH);
        Uri uri = Uri.fromFile(file);

        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"ExoPlayer"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
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

    private class ExoPlayerHandler extends Handler {
        @Override
        public void handleMessage(@NonNull Message msg) {
            switch (msg.what) {
                case MESSAGE_EXO_STOP:
                    mSimpleExoPlayer.stop(true);
                    finish();
                    break;
                default:
                    break;
            }
        }
    }
}