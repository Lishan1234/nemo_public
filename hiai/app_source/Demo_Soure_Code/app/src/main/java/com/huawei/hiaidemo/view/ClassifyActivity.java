package com.huawei.hiaidemo.view;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.huawei.hiaidemo.R;
import com.huawei.hiaidemo.bean.ModelInfo;
import com.huawei.hiaidemo.utils.ModelManager;
import com.huawei.hiaidemo.utils.Untils;

import java.io.File;
import java.util.Random;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;
import static com.huawei.hiaidemo.utils.Constant.AI_OK;
import static com.huawei.hiaidemo.utils.Constant.meanValueOfBlue;
import static com.huawei.hiaidemo.utils.Constant.meanValueOfGreen;
import static com.huawei.hiaidemo.utils.Constant.meanValueOfRed;


public class ClassifyActivity extends AppCompatActivity implements View.OnClickListener{

    private static final String TAG = SyncClassifyActivity.class.getSimpleName();

    protected ModelInfo demoModelInfo = new ModelInfo();
    protected RecyclerView rv;
    protected boolean useNPU  = false;
    protected boolean interfaceCompatible = true;
    protected Button btnsync = null;
    protected Button btnasync = null;
    protected LinearLayoutManager manager = null;
    protected float[][] outputData;
    protected float inferenceTime;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classify);
        initView();
        initModels();
        copyModels();
        modelCompatibilityProcess(); //+Load SO files

        //Run benchmark
        loadModelFromFile(demoModelInfo.getOfflineModelName(),demoModelInfo.getModelSaveDir()+demoModelInfo.getOfflineModel(),demoModelInfo.isMixModel());
        float[][] input_data = prepareData();
        runModel(demoModelInfo, input_data);
    }

    public static float[] NHWCtoNCHW(float[] orinal,int N, int C,int H,int W){
        if(orinal == null || orinal.length == 0 || N*H*W*C == 0 || N < 0 || C < 0 || H < 0 || W < 0){
            return orinal;
        }
        float[] nchw = new float[orinal.length];
        for(int i = 0; i < N ;i++){
            for(int j = 0; j < C ;j++){
                for(int k = 0; k < H*W ; k++){
                    nchw[i*C*H*W+j*H*W+k] = orinal[i*H*W*C+ k*C + j];
                }
            }
        }
        return nchw;
    }

    protected float[] generateFakeFrame(int height, int width, int channel){
        float[] buff = new float[channel * width * height];
        Random rand = new Random();

        int batch = 1;
        int k = 0;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {

                buff[k] = (float) (rand.nextInt(255)+1) / 255;
                k++;
                buff[k] = (float) (rand.nextInt(255)+1) / 255;
                k++;
                buff[k] = (float) (rand.nextInt(255)+1) / 255;
                k++;
            }
        }

        return NHWCtoNCHW(buff, batch, channel, height, width);
    }

    protected float[][] prepareData(){
        Log.i(TAG,"Generate a fake frame");

        float[] inputData = generateFakeFrame(demoModelInfo.getInput_H(), demoModelInfo.getInput_W(), demoModelInfo.getInput_C());
        float inputdatas[][] = new float[1][];
        inputdatas[0] = inputData;
        return inputdatas;
    }

    protected void loadModelFromFile(String offlineModelName, String offlineModelPath,boolean isMixModel) {
        int ret = ModelManager.loadModelFromFileSync(offlineModelName,offlineModelPath,isMixModel);
        if (AI_OK == ret) {
            Log.i(TAG, "Model load success");
        } else {
            Log.i(TAG, "Model load fail");
        }
    }

    protected void runModel(ModelInfo modelInfo, float[][] inputData) {
        long start = System.currentTimeMillis();
        outputData = ModelManager.runModelSync(modelInfo, inputData);
        long end = System.currentTimeMillis();
        inferenceTime = end - start;
        if(outputData == null){
            Log.e(TAG,"runModelSync fail ,outputData is null");
            return;
        }
        Log.i(TAG, "runModel outputdata length : " + outputData.length + "/inferenceTime = "+inferenceTime);

        //postProcess(outputData);
    }

    private void initView() {
        manager = new LinearLayoutManager(this);
        btnsync = (Button) findViewById(R.id.btn_sync);
        btnasync = (Button) findViewById(R.id.btn_async);
        btnsync.setOnClickListener(this);
        btnasync.setOnClickListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.btn_sync:
                if(interfaceCompatible) {
                    if (useNPU) {
                        Intent intent = new Intent(ClassifyActivity.this, SyncClassifyActivity.class);
                        intent.putExtra("modelInfo", demoModelInfo);
                        startActivity(intent);
                    } else {
                        Toast.makeText(this, "Model incompatibility or NO online Compiler interface or Compile model failed, Please run it on CPU", Toast.LENGTH_SHORT).show();
                    }
                }else {
                    Toast.makeText(this, "Interface incompatibility, Please run it on CPU", Toast.LENGTH_SHORT).show();
                }
                break;

            case R.id.btn_async:
                if(interfaceCompatible) {
                    if (useNPU) {
                        Intent intent = new Intent(ClassifyActivity.this, AsyncClassifyActivity.class);
                        intent.putExtra("modelInfo", demoModelInfo);
                        startActivity(intent);
                    } else {
                        Toast.makeText(this, "Model incompatibility or NO online Compiler interface or Compile model failed, Please run it on CPU", Toast.LENGTH_SHORT).show();
                    }
                }else{
                    Toast.makeText(this, "Interface incompatibility, Please run it on CPU", Toast.LENGTH_SHORT).show();
                }
                break;
        }

    }
    private void copyModels(){
        AssetManager am = getAssets();
        if(!Untils.isExistModelsInAppModels(demoModelInfo.getOnlineModel(),demoModelInfo.getModelSaveDir())){
            Untils.copyModelsFromAssetToAppModelsByBuffer(am, demoModelInfo.getOnlineModel(),demoModelInfo.getModelSaveDir());
        }
        if(!Untils.isExistModelsInAppModels(demoModelInfo.getOnlineModelPara(),demoModelInfo.getModelSaveDir())){
            Untils.copyModelsFromAssetToAppModelsByBuffer(am, demoModelInfo.getOnlineModelPara(),demoModelInfo.getModelSaveDir());
        }
        if(!Untils.isExistModelsInAppModels(demoModelInfo.getOfflineModel(),demoModelInfo.getModelSaveDir())){
            Untils.copyModelsFromAssetToAppModelsByBuffer(am, demoModelInfo.getOfflineModel(),demoModelInfo.getModelSaveDir());
        }
    }

    private void modelCompatibilityProcess(){
        //load libhiaijni.so
        boolean isSoLoadSuccess = ModelManager.loadJNISo();

        if (isSoLoadSuccess) {//npu
            Toast.makeText(this, "load libhiai.so success.", Toast.LENGTH_SHORT).show();

            interfaceCompatible = true;
            useNPU = ModelManager.modelCompatibilityProcessFromFile(demoModelInfo.getModelSaveDir() + demoModelInfo.getOnlineModel(),
                    demoModelInfo.getModelSaveDir() + demoModelInfo.getOnlineModelPara(),
                    demoModelInfo.getFramework(),demoModelInfo.getModelSaveDir() + demoModelInfo.getOfflineModel(),demoModelInfo.isMixModel());

//            byte[] onlinemodebuffer = Untils.getModelBufferFromModelFile(demoModelInfo.getModelSaveDir() + demoModelInfo.getOnlineModel());
//            byte[] onlinemodeparabuffer = Untils.getModelBufferFromModelFile(demoModelInfo.getModelSaveDir() + demoModelInfo.getOnlineModelPara());
//            useNPU = ModelManager.modelCompatibilityProcessFromBuffer(onlinemodebuffer,onlinemodeparabuffer,demoModelInfo.getFramework(),
//                    demoModelInfo.getModelSaveDir()+demoModelInfo.getOfflineModel(),demoModelInfo.isMixModel());
        }
        else {
            interfaceCompatible = false;
            Toast.makeText(this, "load libhiai.so fail.", Toast.LENGTH_SHORT).show();
        }
    }
    protected void initModels(){
        File dir =  getDir("models", Context.MODE_PRIVATE);
        String path = dir.getAbsolutePath() + File.separator;


        demoModelInfo.setModelSaveDir(path);
        //demoModelInfo.setOnlineModel("squeezenet_deploy_quant8.prototxt");
        //demoModelInfo.setOnlineModelPara("squeezenet_v1.1.caffemodel");
        demoModelInfo.setOnlineModel("final_240_426_3.txt");
        demoModelInfo.setOnlineModelPara("final_240_426_3.pb");
        //demoModelInfo.setFramework("caffe_8bit");
        demoModelInfo.setFramework("tensorflow");
        //demoModelInfo.setOfflineModel("offline_squeezenet");
        //demoModelInfo.setOfflineModelName("squeezenet");
        demoModelInfo.setOfflineModel("final_240_426_3.cambricon");
        demoModelInfo.setOfflineModelName("final_240_426_3");
        demoModelInfo.setMixModel(false);
        demoModelInfo.setInput_Number(1);
        demoModelInfo.setInput_N(1);
        demoModelInfo.setInput_C(3);
        //demoModelInfo.setInput_H(227);
        //demoModelInfo.setInput_W(227);
        demoModelInfo.setInput_H(240);
        demoModelInfo.setInput_W(426);
        demoModelInfo.setOutput_Number(1);
        demoModelInfo.setOutput_N(1);
        demoModelInfo.setOutput_C(3);
        demoModelInfo.setOutput_H(960);
        demoModelInfo.setOutput_W(1706);
        //demoModelInfo.setOutput_N(1);
        //demoModelInfo.setOutput_C(1000);
        //demoModelInfo.setOutput_H(1);
        //demoModelInfo.setOutput_W(1);
        //demoModelInfo.setOnlineModelLabel("labels_squeezenet.txt");
    }

}
