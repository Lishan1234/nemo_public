package com.huawei.hiaidemo.view;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;

import com.huawei.hiaidemo.R;
import com.huawei.hiaidemo.bean.ModelInfo;
import com.huawei.hiaidemo.utils.Loader;
import com.huawei.hiaidemo.utils.Model;
import com.huawei.hiaidemo.utils.ModelManager;
import com.huawei.hiaidemo.utils.Untils;
import com.huawei.hiaidemo.utils.RawExtracter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;
import static com.huawei.hiaidemo.deprecated.Constant.AI_OK;

public class ClassifyActivity extends AppCompatActivity{

    private static final String TAG = ClassifyActivity.class.getSimpleName();

    protected ModelInfo demoModelInfo = new ModelInfo();
    protected float[][] outputData;
    protected float inferenceTime;

    private int getRawResourceId(String rawName) {
        return this.getResources().getIdentifier(rawName, "raw", this.getPackageName());
    }

    protected void loadModelFromFile(String offlineModelName, String offlineModelPath,boolean isMixModel) {
        int ret = ModelManager.loadModelFromFileSync(offlineModelName,offlineModelPath,isMixModel);
        if (AI_OK == ret) {
            Log.i(TAG, "Model load success");
        } else {
            Log.i(TAG, "Model load fail");
        }
    }

    private static float[][] loadBitmap(Bitmap bitmap, int Width, int Height) {
        int batch = 1;
        int channel = 3;
        float[] buff = new float[channel * Width * Height];

        int k = 0;
        for (int i = 0; i < Height; i++) {
            for (int j = 0; j < Width; j++) {

                int color = bitmap.getPixel(j, i);

                //NHWC
                buff[k] = (float) (red(color))/255;
                k++;
                buff[k] = (float) (green(color))/255;
                k++;
                buff[k] = (float) (blue(color))/255;
                k++;
            }
        }


        float inputdatas[][] = new float[1][];
        inputdatas[0] = Untils.NHWCtoNCHW(buff,batch,channel,Height,Width);

        return inputdatas;
    }

    private Bitmap convertToBitmap(float[][] outputData, int n, int c , int h, int w)
    {
        float[] outputdata_float = outputData[0];
        int[] outputdata = new int [outputdata_float.length];

        for(int i = 0; i < outputdata_float.length; i++) {
            if (outputdata_float[i] > 1) {
                outputdata_float[i] = 1;
            }
            if (outputdata_float[i] < 0) {
                outputdata_float[i] = 0;
            }

            outputdata[i] = Math.round(outputdata_float[i] * 255);
        }

        outputdata = Untils.NCHWtoNHWC(outputdata, n, c, h, w);
        Bitmap bitmap = Bitmap.createBitmap(outputdata, w, h, Bitmap.Config.ARGB_8888);
        //Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        //bitmap.setPixels(outputdata, 0, w, 0, 0, w, h);

        try (FileOutputStream out = new FileOutputStream(demoModelInfo.getModelSaveDir()+"debug.png")) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap;
    }

    double calculatePSNR(float[][] arr1_, float[][] arr2_, int height, int width)
    {
        float[] arr1 = arr1_[0];
        float[] arr2 = arr2_[0];

        if (arr1.length != arr2.length)
        {
            throw new RuntimeException("arrays were of diffrent size");
        }

        final int numPixels = width * height * 3;
        double noise = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y + x * height) * 3;
                noise += (arr1[idx] - arr2[idx]) * (arr1[idx] - arr2[idx]);
                noise += (arr1[idx+1] - arr2[idx+1]) * (arr1[idx+1] - arr2[idx+1]);
                noise += (arr1[idx+2] - arr2[idx+2]) * (arr1[idx+2] - arr2[idx+2]);
            }
        }

        final double mse = noise / numPixels;
        final double psnr = 20 * Math.log10(1.0) - 10 * Math.log10(mse);

        Log.i(TAG, String.format("psnr: %f", psnr));

        return psnr;
    }

    protected void initModels(final String modelName){
        File dir  = getExternalFilesDir("models/edsr");
        String path = dir.getAbsolutePath() + File.separator;

        demoModelInfo.setModelSaveDir(path);
        demoModelInfo.setFramework("tensorflow");
        demoModelInfo.setOfflineModel(modelName + ".cambricon");
        demoModelInfo.setOfflineModelName(modelName);
        demoModelInfo.setMixModel(false);
        demoModelInfo.setInput_Number(1);
        demoModelInfo.setInput_N(1);
        demoModelInfo.setInput_C(3);
        demoModelInfo.setInput_H(240);
        demoModelInfo.setInput_W(426);
        demoModelInfo.setOutput_Number(1);
        demoModelInfo.setOutput_N(1);
        demoModelInfo.setOutput_C(3);
        demoModelInfo.setOutput_H(960);
        demoModelInfo.setOutput_W(1704);
    }

    @Deprecated
    double calculatePSNR(Bitmap bitmap1, Bitmap bitmap2, int height, int width)
    {
        if (bitmap1.getWidth() != bitmap2.getWidth() ||
                bitmap1.getHeight() != bitmap2.getHeight()) {
            Log.e(TAG, String.format("bitmap1 h: %d, w: %d", bitmap1.getHeight(), bitmap1.getWidth()));
            Log.e(TAG, String.format("bitmap2 h: %d, w: %d", bitmap2.getHeight(), bitmap2.getWidth()));
            throw new RuntimeException("images were of diffrent size");
        }

        if (bitmap1.sameAs(bitmap2)) {
            return 100;
        }

        final int numPixels = width * height * 3;
        double noise = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel1 = bitmap1.getPixel(x, y);
                int pixel2 = bitmap2.getPixel(x, y);
                if (pixel1 != pixel2) {
                    noise += (red(pixel1) - red(pixel2)) * (red(pixel1) - red(pixel2));
                    noise += (green(pixel1) - green(pixel2)) * (green(pixel1) - green(pixel2));
                    noise += (blue(pixel1) - blue(pixel2)) * (blue(pixel1) - blue(pixel2));
                }
            }
        }
        final double mse = noise / numPixels;
        final double psnr = 20 * Math.log10(255) - 10 * Math.log10(mse);

        Log.i(TAG, String.format("psnr: %f", psnr));

        return psnr;
    }

    private void saveBitmap(Bitmap bitmap, String name)
    {
        try (FileOutputStream out = new FileOutputStream(demoModelInfo.getModelSaveDir()+name)) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
            // PNG is a lossless format, the compression factor (100) is ignored
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify);

        //Unzip and copy model and image files
        RawExtracter.execute(this, "edsr", getRawResourceId("edsr"));

        //Load a shared library
        boolean isSoLoadSuccess = ModelManager.loadJNISo();
        if (isSoLoadSuccess) {
            Log.i(TAG, "loadJNISo success");
        } else {
            Log.e(TAG, "loadJNISo success");
            this.finish();
            System.exit(0);
        }

        //Iterate over models
        final String modelName = "final_240_426_3";
        final Model model = Loader.createModel(getExternalFilesDir("models/edsr"), modelName);
        initModels(modelName);

        //Load a model
        loadModelFromFile(model.modelName, model.offlineModel.getAbsolutePath(), false);

        //Run a model
        ArrayList<Pair<Double, Float>> results = new ArrayList<>();
        ArrayList<Double> psnr_results = new ArrayList<>();
        ArrayList<Float> runtime_results = new ArrayList<>();
        for (int i = 0; i < model.pngHrImages.length; i++) {
            Bitmap lrBitmap = BitmapFactory.decodeFile(model.pngLrImages[i].getAbsolutePath());
            Bitmap hrBitmap = BitmapFactory.decodeFile(model.pngHrImages[i].getAbsolutePath());

            float[][] inputData = loadBitmap(lrBitmap, demoModelInfo.getInput_W(), demoModelInfo.getInput_H());

            Bitmap lrBitmap_ = convertToBitmap(inputData, demoModelInfo.getInput_N(), demoModelInfo.getInput_C(), demoModelInfo.getInput_H(), demoModelInfo.getInput_W());
            saveBitmap(lrBitmap_, "lr_debug_.png");

            Object[] resp = ModelManager.runModelSync(demoModelInfo, inputData);
            outputData = (float[][])resp[0];
            inferenceTime = (Integer)resp[1];

            //Calculate PSNR
            double psnr = calculatePSNR(outputData, loadBitmap(hrBitmap, demoModelInfo.getOutput_W(), demoModelInfo.getOutput_H()), demoModelInfo.getOutput_H(), demoModelInfo.getOutput_W());
            psnr_results.add(psnr);
            runtime_results.add(inferenceTime);
        }

        double psnr_sum = 0;
        for (Double psnr: psnr_results)
        {
            psnr_sum += psnr;
        }
        double psnr_avg = psnr_sum / psnr_results.size();

        float runtime_sum = 0;
        float runtime_min = -1;
        float runtime_max = -1;
        for (Float runtime: runtime_results)
        {
            runtime_sum += runtime;
            if (runtime_min == -1 || runtime_min > runtime)
            {
                runtime_min = runtime;
            }
            if (runtime_max == -1 || runtime_max < runtime)
            {
                runtime_max = runtime;
            }
        }
        float runtime_avg = runtime_sum / runtime_results.size();

        Log.i(TAG, "psnr_Avg = " + psnr_avg);
        Log.i(TAG, "inferenceTime_Avg = " + runtime_avg);
        Log.i(TAG, "inferenceTime_Min = " + runtime_min);
        Log.i(TAG, "inferenceTime_Max = " + runtime_max);

        results.add(Pair.create(psnr_avg, runtime_avg));
    }
}
