package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class ImageAdjustmentActivity extends AppCompatActivity {
    private SeekBar brightnessSlider, sharpnessSlider, contrastSlider;
    private ImageView imageView;
    private ImageView selectedImageView;
    private Bitmap originalBitmap;
    private Uri imageUri;
    private TextView brightnessValueText;
    private TextView sharpnessValueText;
    private TextView contrastValueText;
    private static final int REQUEST_GALLERY_PICK = 1011;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_adjustment);
        imageView = findViewById(R.id.adjustView);
        brightnessSlider = findViewById(R.id.brightnessSlider);
        sharpnessSlider = findViewById(R.id.sharpnessSlider);
        contrastSlider = findViewById(R.id.contrastSlider);
        Button applyButton = findViewById(R.id.apply_button);
        Button pickImage = findViewById(R.id.pick_gallery_Image);
        brightnessValueText = findViewById(R.id.brightnessValueText);
        contrastValueText = findViewById(R.id.contrastValueText);
        sharpnessValueText = findViewById(R.id.sharpnessValueText);
        // Get the image URI from the intent
        selectedImageView = findViewById(R.id.selectedImageView);
        imageView.setImageBitmap(originalBitmap);
        if (OpenCVLoader.initLocal()) {
            Log.i("OpenCV", "OpenCV successfully loaded.");
        }
        pickImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickImageFromGallery();
            }
        });
        applyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (originalBitmap != null) {
                    int brightnessVal = brightnessSlider.getProgress(); // range 0–100
                    int sharpnessVal = sharpnessSlider.getProgress();   // range 0–100
                    int contrastVal = contrastSlider.getProgress();     // range 0–100
                    // Convert bitmap to Mat
                    Mat mat = new Mat();
                    Utils.bitmapToMat(originalBitmap, mat);

                    // Apply adjustments
                    Mat adjustedMat = adjustImageWithOpenCV(mat, brightnessVal, sharpnessVal, contrastVal);

                    // Convert back to Bitmap
                    Bitmap adjustedBitmap = Bitmap.createBitmap(adjustedMat.cols(), adjustedMat.rows(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(adjustedMat, adjustedBitmap);
                    imageView.setImageBitmap(adjustedBitmap);
                    imageView.setVisibility(View.VISIBLE);
                } else {
                    Toast.makeText(ImageAdjustmentActivity.this, "No image selected!", Toast.LENGTH_SHORT).show();
                }

            }
        });
        brightnessSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                brightnessValueText.setText("Brightness: " + progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user starts dragging
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user stops dragging
            }
        });
        sharpnessSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                sharpnessValueText.setText("Sharpness: " + progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user starts dragging
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user stops dragging
            }
        });
        contrastSlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                contrastValueText.setText("Contrast: " + progress);
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user starts dragging
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // Optional: do something when user stops dragging
            }
        });
    }
    public void checkImage(Bitmap bitmap){
        try {
//            Bitmap resizedBitmap = getResizedBitmapCV(bitmap, 300, 300);
            Mat mat = new Mat();
            org.opencv.android.Utils.bitmapToMat(bitmap, mat);

            // Convert to grayscale
            Mat gray = new Mat();
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);

            // Calculate brightness (mean)
            Scalar meanScalar = Core.mean(gray);
            double brightness = meanScalar.val[0];

            // Calculate sharpness (variance of Laplacian)
            Mat laplacian = new Mat();
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
            Mat laplacianSquared = new Mat();
            Core.multiply(laplacian, laplacian, laplacianSquared);
            Scalar laplacianMean = Core.mean(laplacianSquared);
            double sharpness = laplacianMean.val[0];

            Mat edges = new Mat();
            Imgproc.Canny(gray, edges, 100, 200);
            int edgePixels = Core.countNonZero(edges);

            double edgeDensity = (double) edgePixels / (gray.rows() * gray.cols());
            Core.MinMaxLocResult mmr = Core.minMaxLoc(gray);

            double contrast = mmr.maxVal - mmr.minVal;
            String jsonOutput = String.format("{\"brightness\": %.2f, \"sharpness\": %.2f,\"edgeDensity\": %.2f,\"contrast\": %.2f}", brightness, sharpness,edgeDensity,contrast);
            TextView myTextView = findViewById(R.id.blur_result_text);
            myTextView.setText(jsonOutput);
            myTextView.setVisibility(View.VISIBLE);
            Log.d("JSON objects",jsonOutput);
            // (Optional) You can return this string if calling from another method
            // return jsonOutput;

        } catch (Exception e) {
            Log.e("ImageAnalysis", "Error analyzing image", e);
        }
    }
    private Mat adjustImageWithOpenCV(Mat image, int brightnessVal, int sharpnessVal, int contrastVal) {
        Mat adjusted = new Mat();
        image.copyTo(adjusted);

        // Convert to HSV to adjust brightness
        Imgproc.cvtColor(adjusted, adjusted, Imgproc.COLOR_BGR2HSV);
        java.util.List<Mat> hsvChannels = new java.util.ArrayList<>();
        Core.split(adjusted, hsvChannels);
        double brightnessFactor = (brightnessVal - 50) * 2.0;
        Core.add(hsvChannels.get(2), new Scalar(brightnessFactor), hsvChannels.get(2));
        Core.min(hsvChannels.get(2), new Scalar(255), hsvChannels.get(2));
        Core.max(hsvChannels.get(2), new Scalar(0), hsvChannels.get(2));
        Core.merge(hsvChannels, adjusted);
        Imgproc.cvtColor(adjusted, adjusted, Imgproc.COLOR_HSV2BGR);

        // Contrast adjustment
        double alpha = contrastVal / 50.0; // Range 0 to 2
        adjusted.convertTo(adjusted, -1, alpha, 0);

        // Sharpness adjustment
        float k = sharpnessVal / 100f;
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        kernel.put(0, 0,
                0, -k, 0,
                -k, 1 + 4 * k, -k,
                0, -k, 0);
        Imgproc.filter2D(adjusted, adjusted, adjusted.depth(), kernel);

        return adjusted;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_GALLERY_PICK && data != null) {
                Log.d("data", String.valueOf(data));
                Uri selectedImageUri = data.getData();
                Log.d("selectedImageUri", String.valueOf(selectedImageUri));
                analyzeImage(String.valueOf(selectedImageUri));
                selectedImageView.setImageURI(selectedImageUri);
                selectedImageView.setVisibility(View.VISIBLE);
            }
        }
    }
    public File createTempFileFromUri(Uri uri) throws IOException {
        InputStream inputStream = getContentResolver().openInputStream(uri);
        String fileName = "temp_image_" + System.currentTimeMillis() + ".jpg";
        File tempFile = new File(getCacheDir(), fileName);

        FileOutputStream outputStream = new FileOutputStream(tempFile);

        byte[] buffer = new byte[1024];
        int len;
        while ((len = inputStream.read(buffer)) > 0) {
            outputStream.write(buffer, 0, len);
        }

        outputStream.close();
        inputStream.close();

        return tempFile;
    }

    public void analyzeImage(String imagePath) {
        try {
            File tempFile  = createTempFileFromUri(Uri.parse(imagePath));
            Bitmap bitmap = BitmapFactory.decodeFile(tempFile.getAbsolutePath());
            originalBitmap = bitmap;

        } catch (Exception e) {
            Log.e("ImageAnalysis", "Error analyzing image", e);
        }
    }
    private void pickImageFromGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto, REQUEST_GALLERY_PICK);
    }
}