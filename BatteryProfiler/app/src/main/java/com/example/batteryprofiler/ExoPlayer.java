package com.example.batteryprofiler;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.RequiresApi;
import android.support.constraint.ConstraintLayout;
import android.support.v7.app.AppCompatActivity;
import android.text.PrecomputedText;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import com.google.android.exoplayer2.DefaultLoadControl;
import com.google.android.exoplayer2.DefaultRenderersFactory;
import com.google.android.exoplayer2.ExoPlayerFactory;
import com.google.android.exoplayer2.Player;
import com.google.android.exoplayer2.SimpleExoPlayer;
import com.google.android.exoplayer2.analytics.AnalyticsListener;
import com.google.android.exoplayer2.source.ExtractorMediaSource;
import com.google.android.exoplayer2.source.MediaSource;
import com.google.android.exoplayer2.source.dash.DashChunkSource;
import com.google.android.exoplayer2.source.dash.DashMediaSource;
import com.google.android.exoplayer2.source.dash.DefaultDashChunkSource;
import com.google.android.exoplayer2.trackselection.DefaultTrackSelector;
import com.google.android.exoplayer2.ui.AspectRatioFrameLayout;
import com.google.android.exoplayer2.ui.PlayerView;
import com.google.android.exoplayer2.upstream.DataSource;
import com.google.android.exoplayer2.upstream.DefaultDataSourceFactory;
import com.google.android.exoplayer2.upstream.DefaultHttpDataSourceFactory;
import com.google.android.exoplayer2.util.Util;

public class ExoPlayer extends AppCompatActivity {

    PlayerView mPlayerView;
    SimpleExoPlayer mSimpleExoPlayer;
    BatteryManager mBatteryManager;
    Long start_battery;
    Long end_battery;
    int start_percentage;
    int end_percentage;
    int height;
    int width;

    String video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm";

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.exoplayer);

        //make landscape and hide navigation button
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);

        prep_exo();

        mBatteryManager = (BatteryManager) getApplicationContext().getSystemService(getApplicationContext().BATTERY_SERVICE);

        //get current battery
        start_battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)/1000;
        start_percentage = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
        Log.i("TAGG",start_battery + "mAh");
        Log.i("TAGG",start_percentage + "%");

        //listener for end battery
        mSimpleExoPlayer.addListener(new Player.EventListener(){
            @Override
            public void onPlayerStateChanged(boolean playWhenReady, int playbackState){
                switch(playbackState){
                    case Player.STATE_ENDED:
                        end_battery = mBatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER);
                        end_percentage = mBatteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
                        Log.i("TAGG",end_battery+"mAh");
                        Log.i("TAGG",end_percentage + "%");

                        //finish and go back to previous activity
                        Intent data = new Intent();
                        data.putExtra("start bat",start_battery);
                        data.putExtra("end bat", end_battery);
                        data.putExtra("start percentage",start_percentage);
                        data.putExtra("end percentage", end_percentage);
                        setResult(RESULT_OK,data);
                        finish();
                        break;
                    default:
                        break;
                }
            }
        });
        mSimpleExoPlayer.addAnalyticsListener(new AnalyticsListener() {
            @Override
            public void onVideoSizeChanged(EventTime eventTime, int width, int height, int unappliedRotationDegrees, float pixelWidthHeightRatio) {
                width = width;
                height = height;
                Log.i("TAGG",width+"x"+height);
                Toast toast = Toast.makeText(getApplicationContext(),width+"x"+height,Toast.LENGTH_SHORT);
            }
        });


        mSimpleExoPlayer.setPlayWhenReady(true);
    }

    public void prep_exo(){

        Intent intent = getIntent();
        if(intent.getIntExtra("requestcode",0) == 1) {
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-360.webm";
        }
        else if(intent.getIntExtra("requestcode",0)==2){
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-720.webm";
        }
        else if(intent.getIntExtra("requestcode",0)==3){
            video = "https://storage.googleapis.com/exoplayer-test-media-1/gen-3/screens/dash-vod-single-segment/video-vp9-1080.webm";
        }

        //create player and set it to view
        mPlayerView = findViewById(R.id.video_view);


        DefaultRenderersFactory renderFactory = new DefaultRenderersFactory(this);
        renderFactory.setExtensionRendererMode(DefaultRenderersFactory.EXTENSION_RENDERER_MODE_PREFER);
        mSimpleExoPlayer =
                ExoPlayerFactory.newSimpleInstance(this,
                        renderFactory,
                        new DefaultTrackSelector(),
                        new DefaultLoadControl(),
                        null);
        mPlayerView.setPlayer(mSimpleExoPlayer);

        //stretch to fit screen
        mPlayerView.setResizeMode(AspectRatioFrameLayout.RESIZE_MODE_FILL);

        //Create  media source
        Uri uri = Uri.parse(video);

        //Create normal media source
        MediaSource mediaSource = createMediaSource(uri);

        //use media source
        mSimpleExoPlayer.prepare(mediaSource);
    }

    private MediaSource createMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultDataSourceFactory(this, Util.getUserAgent(this,"vp9testing"));
        MediaSource videoSource = new ExtractorMediaSource.Factory(dataSourceFactory).createMediaSource(uri);
        return videoSource;
    }

    private DashMediaSource createDashMediaSource(Uri uri){
        DataSource.Factory dataSourceFactory = new DefaultHttpDataSourceFactory("ExoPlayer");
        DashChunkSource.Factory dashChunkSourceFactory =
                new DefaultDashChunkSource.Factory(new DefaultHttpDataSourceFactory("ExoPlayer"));
        DashMediaSource mediaSource = new DashMediaSource.Factory(dashChunkSourceFactory, dataSourceFactory).createMediaSource(uri);
        return mediaSource;
    }
}
