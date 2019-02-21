package com.example.chong.vp9testing;

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.content.pm.ApplicationInfo;
import android.media.MediaExtractor;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.PrecomputedText;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.LinearLayout;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.Renderer;
import com.google.android.exoplayer2.RenderersFactory;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.drm.DrmSessionManager;
import com.google.android.exoplayer2.drm.FrameworkMediaCrypto;
import com.google.android.exoplayer2.ext.vp9.LibvpxVideoRenderer;
import com.google.android.exoplayer2.source.ExtractorMediaSource;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.dash.DashChunkSource;
import com.google.android.exoplayer2.source.dash.DashMediaSource;
import com.google.android.exoplayer2.source.dash.DefaultDashChunkSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.upstream.DefaultHttpDataSourceFactory;
import com.google.android.exoplayer2.util.Log;
import com.google.android.exoplayer2.util.Util;
import com.google.android.exoplayer2.video.VideoRendererEventListener;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    PlayerView mPlayerView;
    SimpleExoPlayer mSimpleExoPlayer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

//        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);


        //make new vid renderers factory
//        RenderersFactory renderersFactory = new DefaultRenderersFactory(this, this){
//            @Override
//            protected void buildVideoRenderers(Context context, DrmSessionManager<FrameworkMediaCrypto> drmSessionManager,
//                                               long allowedVideoJoiningTimeMs, Handler eventHandler, VideoRendererEventListener eventListener,
//                                               int extensionRendererMode, ArrayList<Renderer> out){
//                LibvpxVideoRenderer videoRenderer = new LibvpxVideoRenderer(true, 0);
//                out.add(videoRenderer);
//            }
//        };

        //create player and set it to view
        mPlayerView = findViewById(R.id.video_view);
        mSimpleExoPlayer =
                ExoPlayerFactory.newSimpleInstance(this,
                        new DefaultRenderersFactory(this, this,DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER),
                        new DefaultTrackSelector(),
                        new DefaultLoadControl(),
                        null);
        mPlayerView.setPlayer(mSimpleExoPlayer);

        //hide action bar
        getSupportActionBar().hide();
        //full screen
        ViewGroup.LayoutParams params = mPlayerView.getLayoutParams();
        params.width = params.MATCH_PARENT;
        params.height = params.MATCH_PARENT;
        mPlayerView.setLayoutParams(params);


                //Create  media source
        Uri uri = Uri.parse("https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm");
        //https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
        //https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm

        //Create normal media source
        MediaSource mediaSource = createMediaSource(uri);

        //mediaextractor for testing


        //use media source
        mSimpleExoPlayer.prepare(mediaSource);
//        mSimpleExoPlayer.setPlayWhenReady(true);
    }


    private DashMediaSource createDashMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultHttpDataSourceFactory("ExoPlayer");
        DashChunkSource.Factory dashChunkSourceFactory =
                new DefaultDashChunkSource.Factory(new DefaultHttpDataSourceFactory("ExoPlayer"));
        DashMediaSource mediaSource = new DashMediaSource.Factory(dashChunkSourceFactory, dataSourceFactory).createMediaSource(uri);
        return mediaSource;
    }

    private MediaSource createMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"vp9testing"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }
}
