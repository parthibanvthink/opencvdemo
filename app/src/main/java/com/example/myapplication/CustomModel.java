package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CustomModel extends AppCompatActivity {
    private static final String MODEL_PATH = "detect.tflite";
    private static final String LABELMAP_PATH = "raw/labels.txt";
    private static final String IMAGE_PATH = "custom_model_lite/test2.jpg";
    private static final float CONFIDENCE_THRESHOLD = 0.5f;
    private Interpreter interpreter;
    private List<String> labels;
   private static final int INPUT_WIDTH = 320;  // Model input width
    private static final int INPUT_HEIGHT = 320;  // Model input height

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_custom_model);
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed.");
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully.");
        }

        // Load the TensorFlow Lite model
        try {
            interpreter = new Interpreter(loadModelFile("detect.tflite"));
            interpreter.allocateTensors();
            int inputTensors = interpreter.getInputTensorCount();
            Log.d("inputTensors", String.valueOf(inputTensors)) ;
            int outputTensors = interpreter.getOutputTensorCount();
            Log.d("outputTensors", String.valueOf(outputTensors));
            for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
                Tensor outTensor = interpreter.getInputTensor(i);
                int[] shape = outTensor.shape();
                DataType type = outTensor.dataType();
                Log.d("OutputTensor", "Input " + i + " shape: " + Arrays.toString(shape));
                Log.d("OutputTensor", "Input " + i + " type: " + type.name());
            }

            for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
                Tensor outTensor = interpreter.getOutputTensor(i);
                int[] shape = outTensor.shape();
                DataType type = outTensor.dataType();
                Log.d("OutputTensor", "Output " + i + " shape: " + Arrays.toString(shape));
                Log.d("OutputTensor", "Output " + i + " type: " + type.name());
            }


        } catch (IOException e) {
            e.printStackTrace();
        }

        // Load the labels from the label map file
        loadLabels(this);

        Mat imageMat = loadImageFromAssets("custom_model_lite/test2.jpg");
        float[][][][] inputData = preprocessMatInput(imageMat, 320, 320);
        Object[] inputArray = {inputData};
        // Step 1: Allocate output arrays with correct shapes
        float[][] outputClasses = new float[1][10];       // [1, 10]
        float[][][] outputBoxes = new float[1][10][4];    // [1, 10, 4]
        float[] numDetections = new float[1];             // [1]
        float[][] outputScores = new float[1][10];        // [1, 10]

// Step 2: Map them correctly
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputScores);     // classes
        outputMap.put(1, outputBoxes);       // boxes
        outputMap.put(2, numDetections);     // number of detections
        outputMap.put(3, outputClasses);
// Run inference (based on your model’s output)
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);  // You'll define outputBuffer based on output tensor shape
        int detectionCount = (int) numDetections[0];
        boolean found = false;
// This will help you see all class predictions for each detection
        Log.d("Detection", "outputClasses (raw): " + Arrays.toString(outputClasses[0]));

        for (int i = 0; i < detectionCount; i++) {
            float score = outputScores[0][i];
            Log.d("score", String.valueOf(score));
            if (score >= 0.5) {
//                Log.d("outputClasses", Arrays.toString(new float[]{outputClasses[0][i]}));
                int classId = (int) outputClasses[0][i];
                Log.d("classId", String.valueOf(classId));
                String label = classId < labels.size() ? labels.get(classId) : "ID " + classId;

                Log.d("Detection", " - " + label + ": " + String.format("%.2f", score));
                found = true;
            }
        }

        if (!found) {
            Log.d("Detection", "No object detected above the confidence threshold.");
        }

    }
    // Utility function
    private int getArgMax(float[] scores) {
        int maxIdx = 0;
        float maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    private float[][][][] preprocessMatInput(Mat imageMat, int inputWidth, int inputHeight) {
        // Convert BGR to RGB
        Mat rgbMat = new Mat();
        Imgproc.cvtColor(imageMat, rgbMat, Imgproc.COLOR_BGR2RGB);

        // Resize to model input size
        Mat resizedMat = new Mat();
        Imgproc.resize(rgbMat, resizedMat, new Size(inputWidth, inputHeight));

        // Convert to float
        resizedMat.convertTo(resizedMat, CvType.CV_32FC3);

        // Normalize to [-1, 1] like in Python
        Core.subtract(resizedMat, new Scalar(127.5, 127.5, 127.5), resizedMat);
        Core.divide(resizedMat, new Scalar(127.5, 127.5, 127.5), resizedMat);

        // Convert Mat to float[][][][] [1][H][W][3]
        int height = resizedMat.rows();
        int width = resizedMat.cols();
        float[][][][] input = new float[1][height][width][3];

        float[] data = new float[height * width * 3];
        resizedMat.get(0, 0, data);

        int index = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                input[0][y][x][0] = data[index++];
                input[0][y][x][1] = data[index++];
                input[0][y][x][2] = data[index++];
            }
        }

        return input;
    }

    private Mat loadImageFromAssets(String assetPath) {
        try {
            InputStream is = getAssets().open(assetPath);
            Bitmap bitmap = BitmapFactory.decodeStream(is);

            // Convert Bitmap to Mat
            Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
            Utils.bitmapToMat(bitmap, mat);
            return mat;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    // Convert OpenCV Mat to Bitmap (for Android UI display)
    private Bitmap matToBitmap(Mat mat) {
        Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
        org.opencv.android.Utils.matToBitmap(mat, bitmap);
        return bitmap;
    }
    private MappedByteBuffer loadModelFile(String assetPath) throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd(assetPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Load the labels from the label map file
    private void loadLabels(Context context) {
        labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(context.getResources().openRawResource(R.raw.labels)))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("Labels verification", labels.toString());
    }
    // Perform object detection using the TensorFlow Lite interpreter
    private void performObjectDetection(Mat imageMat) {
        Log.d("imageMat in perform objecy detection", String.valueOf(imageMat));
        // Preprocess image and convert to the format the model expects
        float[][][][] inputData = preprocessImage(imageMat);

        // Create output arrays for model inference results
        float[][] outputScores = new float[1][10];  // Modify size according to your model
        float[][][] outputBoxes = new float[1][10][4];  // Modify size according to your model
        float[][] outputClasses = new float[1][10];  // Modify size according to your model
        float[] numDetections = new float[1];

        // Run inference (equivalent to interpreter.invoke())
        interpreter.run(inputData, new Object[]{outputScores, outputBoxes, numDetections, outputClasses});

        // Process the results
        int numDetectionsResult = (int) numDetections[0];

        Log.d("Object Detection", "Number of detections: " + numDetectionsResult);
        for (int i = 0; i < numDetectionsResult; i++) {
            if (outputScores[0][i] >= CONFIDENCE_THRESHOLD) {
                int classId = (int) outputClasses[0][i];
                String label = labels.get(classId);
                Log.d("Detected Object", label + ": " + outputScores[0][i]);
            }
        }
    }

    // Preprocess image for TensorFlow Lite model
    private float[][][][] preprocessImage(Mat imageMat) {
        Log.d("imageMat in preprocessImage", String.valueOf(imageMat));
        // Convert imageMat to the required input shape (1, 320, 320, 3)
        int[] dims = {1, INPUT_WIDTH, INPUT_HEIGHT, 3};
        float[][][][] inputData = new float[1][INPUT_WIDTH][INPUT_HEIGHT][3];
        Log.d("inputData in the inout dataA", Arrays.toString(inputData));
        // Fill inputData with image data
        for (int i = 0; i < INPUT_HEIGHT; i++) {
            for (int j = 0; j < INPUT_WIDTH; j++) {
                double[] pixel = imageMat.get(i, j);
                if (pixel != null && pixel.length >= 3) {
                    inputData[0][i][j][0] = (float) pixel[0];  // B
                    inputData[0][i][j][1] = (float) pixel[1];  // G
                    inputData[0][i][j][2] = (float) pixel[2];  // R
                } else {
                    inputData[0][i][j][0] = 0;
                    inputData[0][i][j][1] = 0;
                    inputData[0][i][j][2] = 0;
                }
//                inputData[0][i][j][0] = (float) pixel[0];  // BGR values
//                inputData[0][i][j][1] = (float) pixel[1];
//                inputData[0][i][j][2] = (float) pixel[2];
            }
        }

        return inputData;
    }
}
