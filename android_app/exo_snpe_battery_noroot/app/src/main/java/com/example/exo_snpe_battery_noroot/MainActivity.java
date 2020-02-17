package com.example.exo_snpe_battery_noroot;

import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.BatteryManager;
import android.os.Bundle;
import android.view.WindowManager;

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

public class MainActivity extends AppCompatActivity {


    public native long jniFunction(int minutes, String dlc, String input, String output, String log, boolean doLog);

    static{
        System.loadLibrary("snpeJNI");
    }


    PlayerView mPlayerView;
    SimpleExoPlayer mSimpleExoPlayer;
    BatteryManager mBatteryManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        prepareExoPlayer();
    }

    private void prepareExoPlayer(){
        mPlayerView = findViewById(R.id.video_view);
        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        mSimpleExoPlayer = ExoPlayerFactory.newSimpleInstance(this,
                renderFactory,
                new DefaultTrackSelector(),
                new DefaultLoadControl(),
                null);
        mSimpleExoPlayer.setRepeatMode(Player.REPEAT_MODE_ALL);
        mPlayerView.setPlayer(mSimpleExoPlayer);
        mPlayerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FIXED_HEIGHT);
        mSimpleExoPlayer.prepare(createLocalMediaSource());
    }

    private MediaSource createLocalMediaSource(){
        String video ="sdcard/BatteryTesting/video/240p_s0_d60_encoded.webm";
        File file = new File(video);
        Uri uri = Uri.fromFile(file);

        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"ExoPlayer"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }
}
