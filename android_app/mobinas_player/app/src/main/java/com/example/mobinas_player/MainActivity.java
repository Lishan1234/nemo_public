package com.example.mobinas_player;

import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Bundle;
import android.view.View;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.ProgressiveMediaSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.AspectRatioFrameLayout;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

import java.io.File;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

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
    }

    private MediaSource createLocalMediaSource(){
        String video = "/storage/emulated/0/Android/data/android.example.testlibvpx/files/video/240p_s0_d60_encoded.webm";
        File file = new File(video);
        Uri uri = Uri.fromFile(file);

        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"ExoPlayer"));
        MediaSource videoSource = new ProgressiveMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }
}
