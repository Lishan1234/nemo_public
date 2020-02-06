package com.example.exo_snpe_battery;

import android.os.Environment;

public class Constants {
    public static String batterySysFile = "";

    public static final int START_THRESHOLD = 2000;

    public static final int ONE_MINUTE_MS = 60000;
    public static final int COOLDOWN_DURATION_MS = 60000 * 10;

    public static final int MESSAGE_NO_BAT = 99;
    public static final int MESSAGE_KEEP_CHARGING = 100;
    public static final int MESSAGE_LOG_BATTERY = 101;
    public static final int MESSAGE_NEXT_TEST = 102;
    public static final int MESSAGE_NEXT_REPEAT = 103;


    public static String videoPath = "sdcard/BatteryTesting/video/240p_s0_d60_encoded.webm";

    public static String lowDlc = "sdcard/BatteryTesting/models/low.dlc";
    public static String mediumDlc = "sdcard/BatteryTesting/models/medium.dlc";
    public static String highDlc = "sdcard/BatteryTesting/models/high.dlc";

    public static String input = "sdcard/BatteryTesting/input/inputs";
    public static String output =  "sdcard/BatteryTesting/output";
    public static int minutes = 1;

    public enum TestType{
        LOW,MEDIUM,HIGH,NONE
    }
}

