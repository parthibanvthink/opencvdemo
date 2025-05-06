package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import com.google.gson.Gson;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class OpenAIClip extends AppCompatActivity {

    private static final String TAG = "CLIPModel";
    private Interpreter tflite;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_open_aiclip);
        try {
            // Load model
            tflite = new Interpreter(loadModel("openai_clip.tflite"));

            // Load image and preprocess
            float[][][][] imageTensor = preprocessImage("catdog.jpg");

            // Load tokenized text embedding
            int[][] textEmbedding = loadTextEmbedding("class3_embeddings.json", "cat");

            // Output buffer (modify size as per your model's output shape)
            float[][] output = new float[1][1]; // Assuming output is 1x512

            // Run inference
            Object[] inputs = { imageTensor, textEmbedding };
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, output);

            tflite.runForMultipleInputsOutputs(inputs, outputs);

            Log.d(TAG, "Model Output[0][0]: " + output[0][0]);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModel(String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[][][][] preprocessImage(String imageName) throws Exception {
        InputStream is = getAssets().open(imageName);
        Bitmap bitmap = BitmapFactory.decodeStream(is);
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        float[][][][] input = new float[1][224][224][3];
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int pixel = resized.getPixel(x, y);
                float r = ((pixel >> 16) & 0xFF) / 127.5f - 1.0f;
                float g = ((pixel >> 8) & 0xFF) / 127.5f - 1.0f;
                float b = (pixel & 0xFF) / 127.5f - 1.0f;
                input[0][y][x][0] = r;
                input[0][y][x][1] = g;
                input[0][y][x][2] = b;
            }
        }
        return input;
    }

//    private int[][] loadTextEmbedding(String filename, String label) throws Exception {
//        InputStream is = getAssets().open(filename);
//        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
//        Gson gson = new Gson();
//        Map<String, float[]> embeddings = gson.fromJson(reader, Map.class);
//
//        Object array = embeddings.get(label);
//        if (array instanceof java.util.ArrayList) {
//            java.util.ArrayList<?> list = (java.util.ArrayList<?>) array;
//            int[][] result = new int[1][list.size()];
//            for (int i = 0; i < list.size(); i++) {
//                result[0][i] = ((Number) list.get(i)).intValue();
//            }
//            return result;
//        }
//
//        throw new Exception("Invalid embedding format");
//    }
private int[][] loadTextEmbedding(String filename, String label) throws Exception {
    InputStream is = getAssets().open(filename);
    BufferedReader reader = new BufferedReader(new InputStreamReader(is));
    Gson gson = new Gson();

    Map<String, ArrayList<Double>> tokens = gson.fromJson(reader, Map.class);
    ArrayList<Double> tokenList = tokens.get(label);

    if (tokenList == null || tokenList.size() != 77) {
        throw new Exception("Token array missing or wrong size for label: " + label);
    }

    int[][] result = new int[1][77];
    for (int i = 0; i < 77; i++) {
        result[0][i] = tokenList.get(i).intValue(); // cast safely
    }
    return result;
}

}