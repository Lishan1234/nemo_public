package com.example.exo_snpe_battery;

public class BatteryTestUnit {

    Constants.TestType testType;
    int minutes;
    int repeat;
    boolean log;


    BatteryTestUnit(Constants.TestType testType, int repeat, int minutes, boolean log){
        this.testType = testType;
        this.repeat = repeat;
        this.minutes = minutes;
        this.log = log;
    }
}
