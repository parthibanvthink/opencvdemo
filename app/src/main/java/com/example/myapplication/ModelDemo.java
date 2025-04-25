package com.example.myapplication;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.opengl.Visibility;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;
import java.util.Set;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class ModelDemo extends AppCompatActivity {
    private Interpreter tflite;
    private static final String MODEL_PATH = "1.tflite";


    private EditText inputText;
    private Button classifyButton;
    private TextView resultText;
    private static final int PICK_IMAGE = 1001;
    ImageView imageView;
    TextView predictionText;
    OrtEnvironment env;
    OrtSession session;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_model_demo);
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Initialize UI elements
        inputText = findViewById(R.id.inputText);
        classifyButton = findViewById(R.id.classifyButton);
        resultText = findViewById(R.id.resultText);
        classifyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String input = inputText.getText().toString();
                if (!input.isEmpty()) {
                    classifyText(input);
                } else {
                    Toast.makeText(ModelDemo.this, "Please enter some text", Toast.LENGTH_SHORT).show();
                }
            }
        });
        imageView = findViewById(R.id.imageView);
        predictionText = findViewById(R.id.predictionText);
        Button selectImageBtn = findViewById(R.id.selectImageBtn);

        try {
            env = OrtEnvironment.getEnvironment();
            session = loadONNXModel("blur_classification.onnx");

//            session = env.createSession(getAssets().openFd("blur_classification.onnx").getFileDescriptor().toString(), new OrtSession.SessionOptions());
        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show();
        }

        selectImageBtn.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, PICK_IMAGE);
        });
    }
    private String copyModelToInternalStorage(String fileName) {
        File file = new File(getFilesDir(), fileName);
        if (!file.exists()) {
            try (InputStream inputStream = getAssets().open(fileName);
                 FileOutputStream outputStream = new FileOutputStream(file)) {
                byte[] buffer = new byte[1024];
                int length;
                while ((length = inputStream.read(buffer)) > 0) {
                    outputStream.write(buffer, 0, length);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return file.getAbsolutePath();
    }

    private OrtSession loadONNXModel(String assetName) {
        try {
            // Step 1: Copy model from assets to cache dir
            InputStream inputStream = getAssets().open(assetName);
            java.io.File file = new java.io.File(getCacheDir(), assetName);
            java.io.FileOutputStream outputStream = new java.io.FileOutputStream(file);

            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            outputStream.flush();
            outputStream.close();
            inputStream.close();

            // Step 2: Load ONNX model
            return env.createSession(file.getAbsolutePath(), new OrtSession.SessionOptions());

        } catch (Exception e) {
            e.printStackTrace();
            Toast.makeText(this, "Model load failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
            return null;
        }
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Classify the input text
    private void classifyText(String text) {
        int[] input = tokenizeText(text);
        Log.d("input classify text", Arrays.toString(input));
        int[][] inputArray = new int[1][20]; // model expects shape [1, 20]
        inputArray[0] = input;
        Log.d("inputArray", Arrays.toString(inputArray));
        float[][] outputArray = new float[1][2]; // model outputs shape [1, 2]
        Log.d("outputArray", Arrays.toString(outputArray));
// Now run inference
        tflite.run(inputArray, outputArray);
        Log.d("outputArray afyter result", Arrays.toString(outputArray));
// Output
        Log.d("ModelOutput", "Not spam: " + outputArray[0][0] + ", Spam: " + outputArray[0][1]);
        String resutlt = "Not spam: " + outputArray[0][0] + ", Spam: " + outputArray[0][1];
        resultText.setText(resutlt);
        resultText.setVisibility(View.VISIBLE);
    }
    private int[] tokenizeText(String text) {
        int vocabSize = 2000;
        int[] tokens = new int[20]; // always 20 tokens

        String[] words = text.split("\\s+");

        for (int i = 0; i < 20; i++) {
            if (i < words.length) {
                int token = Math.abs(words[i].hashCode()) % vocabSize;
                tokens[i] = token;
            } else {
                tokens[i] = 0; // pad with 0 if sentence is too short
            }
        }

        return tokens;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            try {
                InputStream imageStream = getContentResolver().openInputStream(imageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(imageStream);
                imageView.setImageBitmap(bitmap);
                imageView.setVisibility(View.VISIBLE);
                runInference(bitmap);
            } catch (Exception e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to process image", Toast.LENGTH_SHORT).show();
            }
        }
    }
public void runInference(Bitmap bitmap) {
    try {
        float[] inputData = preprocessBitmap(bitmap);
        String modelPath = copyModelToInternalStorage("blur_classification.onnx");
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions());

        String inputName = session.getInputNames().iterator().next();
        long[] inputShape = new long[]{1, 3, 224, 224};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), inputShape);

        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

        OnnxValue outputValue = result.get(0);
        Log.d("outputValue", outputValue.toString());
        // Get output names
        Set<String> outputNames = session.getOutputNames();
        Log.d("outputNames", outputNames.toString());
        String outputName = session.getOutputNames().iterator().next(); // Get first output name
        Log.d("outputName",outputName);
        Optional<OnnxValue> optionalValue = result.get(outputName);

        if (optionalValue.isPresent()) {
            OnnxValue value = optionalValue.get();
            Log.d("First if", value.toString());
            if (value instanceof OnnxTensor) {
                OnnxTensor tensor = (OnnxTensor) value;
                Log.d("Second if", tensor.toString());
                // Inspect shape and data
                long[] shape = tensor.getInfo().getShape();
                System.out.println("Shape: " + Arrays.toString(shape));

                float[][] outputArray = (float[][]) tensor.getValue();
                System.out.println("Output: " + Arrays.toString(outputArray[0]));
            }
        } else {
            System.out.println("No output found for: " + outputName);
        }
        float[][] outputArray = (float[][]) outputValue.getValue();
        Log.d("outputArray", Arrays.toString(outputArray));
        float[] prediction = outputArray[0];
        Log.d("prediction", Arrays.toString(prediction));
        float blurProb = prediction[0];
        float notBlurProb = prediction[1];
        double[] logits = {prediction[0], prediction[1]};

// Step 1: Exponentiate (to get positive values)
        double exp0 = Math.exp(logits[0]);
        double exp1 = Math.exp(logits[1]);

// Step 2: Sum of exponentials
        double sum = exp0 + exp1;

// Step 3: Softmax probabilities
        double prob0 = exp0 / sum; // blur
        double prob1 = exp1 / sum; // not blur

        System.out.println("Blur probability: " + prob1);
        System.out.println("Not blur probability: " + prob0);

        String label = prob1 > prob0 ? "Blur Image" : "Not Blur Image";
        predictionText.setText(label);
        predictionText.setVisibility(View.VISIBLE);
        Log.d("ONNX_RESULT", "Prediction: " + label);

    } catch (Exception e) {
        Log.e("ONNX_ERROR", "Error running inference", e);
    }
}
    private float[] preprocessBitmap(Bitmap bitmap) {
        // Resize to 224x224
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();

        float[] inputData = new float[1 * 3 * width * height]; // [1, 3, 224, 224]

        int[] pixels = new int[width * height];
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        // Normalize and convert to CHW format (channel first)
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];

            int r = (pixel >> 16) & 0xFF;
            int g = (pixel >> 8) & 0xFF;
            int b = pixel & 0xFF;

            // Normalize to [0, 1]
            inputData[i] = r / 255.0f;
            inputData[i + width * height] = g / 255.0f;
            inputData[i + 2 * width * height] = b / 255.0f;
        }

        return inputData;
    }

}