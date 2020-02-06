package com.example.exo_snpe_temperature;

import android.content.Context;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;

import androidx.annotation.Nullable;
public class TemperaturePointer extends FrameLayout {
    float dX;
    float dY;
    int id;
    int boundLeft;
    int boundRight;
    int boundTop;
    int boundDown;
    int boundWidth;
    int boundHeight;
    public TemperaturePointer(Context context) {
        super(context);
        init(context);
    }
    public TemperaturePointer(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }
    public TemperaturePointer(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        init(context);
    }
    private void init(Context context) {
        inflate(context, R.layout.temperature_pointer, this);
        boundLeft = 0;
        boundTop = 0;
        boundRight = Integer.MAX_VALUE;
        boundDown = Integer.MAX_VALUE;
        this.setX(0);
        this.setY(0);
    }
    public void setMovableBoundary(int left, int top, int right, int down, int width, int height) {
        boundLeft = left;
        boundTop = top;
        boundRight = right;
        boundDown = down;
        boundWidth = width;
        boundHeight = height;
    }
    public float getHeightRatio() {
        return (getY() - boundTop) / boundHeight;
    }
    public float getWidthRatio() {
        return (getX() - boundLeft) / boundWidth;
    }
    public float getHeightRatioCenter() {
        return (getY() + getHeight()/2 - boundTop) / boundHeight;
    }
    public float getWidthRatioCeter() {
        return (getX() + getWidth()/2 - boundLeft) / boundWidth;
    }

    public float getHeightAdjusted(){
        return getY() + ((float)getHeight())/2;
    }
    public float getWidthAdjusted(){
        return getX() + ((float)getWidth())/2;
    }
    public void setTemperatureText(String str) {
        ((TextView) findViewById(R.id.spotMeterValue)).setText(String.format("#%d\n\n%s", id, str));
    }
    public void setId(int id){
        this.id = id;
        setTemperatureText("");
    }
    public int getTpId(){
        return this.id;
    }
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                dX = this.getX() - event.getRawX();
                dY = this.getY() - event.getRawY();
                break;
            case MotionEvent.ACTION_MOVE:
                float newX = event.getRawX() + dX;
                float newY = event.getRawY() + dY;
//                newX = newX < boundLeft ? boundLeft : newX;
//                newY = newY < boundTop ? boundTop : newY;
//                newX = newX > boundRight - getWidth() ? boundRight - getWidth() : newX;
//                newY = newY > boundDown - getHeight() ? boundDown - getHeight() : newY;
                this.animate()
                        .x(newX)
                        .y(newY)
                        .setDuration(0)
                        .start();
                break;
            default:
                return false;
        }
        Log.d("Pos", String.format("%4.1f | %4.1f", this.getX(), this.getY()));
        return true;
    }
}