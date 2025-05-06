package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;

public class Clip extends AppCompatActivity {
    private static final String TAG = "CLIPTest";
    private static final int IMAGE_SIZE = 224; // assuming 224x224 input
    private static final int EMBEDDING_SIZE = 512; // expected embedding dimension
    private static ArrayList<String> stringList = new ArrayList<>();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_clip);

        try {
            // Load TFLite model
            Interpreter tflite = new Interpreter(loadModelFile("openai_clip.tflite"));

            // Load and preprocess image
            Bitmap bitmap = BitmapFactory.decodeStream(getAssets().open("airplane.jpg"));
            bitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
            ByteBuffer inputBuffer = convertBitmapToByteBuffer(bitmap);

            // Prepare output buffer
            TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
                    new int[]{1, EMBEDDING_SIZE},
                    org.tensorflow.lite.DataType.FLOAT32
            );
            tflite.run(inputBuffer, outputBuffer.getBuffer().rewind());

            // Get image embedding
//            float[] imageEmbedding = outputBuffer.getFloatArray();
            float[] imageEmbedding = normalize(outputBuffer.getFloatArray()); // Normalize image embedding


            // Load class embeddings
            Map<String, float[]> classEmbeddings = loadClassEmbeddings();

            String bestMatch = null;
            float bestScore = -1f;
            float THRESHOLD = 0.25f;

            for (Map.Entry<String, float[]> entry : classEmbeddings.entrySet()) {
                float[] classVec = entry.getValue();
                if (classVec.length != imageEmbedding.length) {
                    Log.w(TAG, "Skipping class: " + entry.getKey() + " due to length mismatch (" + classVec.length + ")");
                    continue;
                }

                float similarity = cosineSimilarity(imageEmbedding, classVec);
//                if (similarity > bestScore) {
                    bestScore = similarity;
                    bestMatch = entry.getKey();
                    String sentence = "Matched class: " + bestMatch + " with similarity: " + bestScore;
                    stringList.add(sentence);
//                }
                if( bestScore >= THRESHOLD){
                    Log.d("BESTSCORE","Matched class: " + bestMatch + " with similarity: " + bestScore);
                }else{
                    Log.d("NO MATCH","Not matched");
                }
            }
            Log.d("stringList", String.valueOf(stringList));
            Log.i(TAG, "Best Match: " + bestMatch + " with similarity: " + bestScore);

        } catch (Exception e) {
            Log.e(TAG, "Error running model", e);
        }
    }

    private MappedByteBuffer loadModelFile(String modelFileName) throws Exception {
        FileInputStream fileInputStream = new FileInputStream(getAssets().openFd(modelFileName).getFileDescriptor());
        FileChannel fileChannel = fileInputStream.getChannel();
        long startOffset = getAssets().openFd(modelFileName).getStartOffset();
        long declaredLength = getAssets().openFd(modelFileName).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 3 * 4); // FLOAT32
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int pixel : intValues) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            byteBuffer.putFloat(r);
            byteBuffer.putFloat(g);
            byteBuffer.putFloat(b);
        }

        return byteBuffer;
    }

    private Map<String, float[]> loadClassEmbeddings() throws Exception {
        InputStream is = getAssets().open("class3_embeddings.json");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder jsonBuilder = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null)
            jsonBuilder.append(line);

        JSONObject json = new JSONObject(jsonBuilder.toString());
        Map<String, float[]> map = new HashMap<>();

        for (Iterator<String> it = json.keys(); it.hasNext(); ) {
            String key = it.next();
            JSONArray arr = json.getJSONArray(key);
            float[] vector = new float[arr.length()];
            for (int i = 0; i < arr.length(); i++)
                vector[i] = (float) arr.getDouble(i);
            map.put(key, vector);
        }
        return map;
    }

    private float cosineSimilarity(float[] vec1, float[] vec2) {
        float dot = 0f, norm1 = 0f, norm2 = 0f;
        for (int i = 0; i < vec1.length; i++) {
            dot += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }
        return dot / (float) (Math.sqrt(norm1) * Math.sqrt(norm2));
    }
    private float[] normalize(float[] vec) {
        float norm = 0f;
        for (float v : vec) norm += v * v;
        norm = (float) Math.sqrt(norm);
        float[] out = new float[vec.length];
        for (int i = 0; i < vec.length; i++) out[i] = vec[i] / (norm + 1e-5f);
        return out;
    }
}
