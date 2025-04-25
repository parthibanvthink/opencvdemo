package com.example.myapplication;

import static android.widget.SeekBar.*;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.myapplication.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_PERMISSIONS = 123;
    private static final int REQUEST_IMAGE_CAPTURE = 456;
    private static final int REQUEST_GALLERY_PICK = 1011;
    private Uri photoUri;
    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;
    ImageView selectedImageView;

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);
        requestPermissions();
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);



        Button galleryButton = findViewById(R.id.pick_gallery_Image);
        selectedImageView = findViewById(R.id.selected_image_view);
        galleryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pickImageFromGallery();
            }
        });

        Button captureButton = findViewById(R.id.open_Camera);
        captureButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openCamera();
            }
        });
        if (OpenCVLoader.initLocal()) {
            Log.i("OpenCV", "OpenCV successfully loaded.");
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
    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        if (cameraIntent.resolveActivity(getPackageManager()) != null) {
            File photoFile = createImageFile();
            if (photoFile != null) {
                photoUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", photoFile);
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
                startActivityForResult(cameraIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }
    private File createImageFile() {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = null;
        try {
            image = File.createTempFile(imageFileName, ".jpg", storageDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }


    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }
    public void analyzeImage(String imagePath) {
        try {
            File tempFile  = createTempFileFromUri(Uri.parse(imagePath));
            Bitmap bitmap = BitmapFactory.decodeFile(tempFile.getAbsolutePath());
//            originalBitmap = bitmap;
            Bitmap resizedBitmap = getResizedBitmapCV(bitmap, 300, 300);
            Mat mat = new Mat();
            org.opencv.android.Utils.bitmapToMat(resizedBitmap, mat);

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
            analyzeImage(sharpness,edgeDensity,contrast,brightness);
            String jsonOutput = String.format("{\"brightness\": %.2f, \"sharpness\": %.2f,\"edgeDensity\": %.2f,\"contrast\": %.2f}", brightness, sharpness,edgeDensity,contrast);
//            TextView myTextView = findViewById(R.id.blur_result_text);
//            myTextView.setText(jsonOutput);
//            myTextView.setVisibility(View.VISIBLE);
            Log.d("JSON objects",jsonOutput);
            // (Optional) You can return this string if calling from another method
            // return jsonOutput;

        } catch (Exception e) {
            Log.e("ImageAnalysis", "Error analyzing image", e);
        }
    }
    private Bitmap getResizedBitmapCV(Bitmap inputBitmap, int newWidth, int newHeight) {
        // Convert the input Bitmap to a Mat
        Mat inputMat = new Mat();
        Utils.bitmapToMat(inputBitmap, inputMat);

        // Create a new Mat for the resized image
        Mat resizedMat = new Mat();
        Imgproc.resize(inputMat, resizedMat, new Size(newWidth, newHeight));

        // Convert the resized Mat back to a Bitmap
        Bitmap resizedBitmap = Bitmap.createBitmap(newWidth, newHeight, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(resizedMat, resizedBitmap);

        inputMat.release();
        resizedMat.release();

        return resizedBitmap;
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


    public void analyzeImage(double sharpness, double edgeDensity, double contrast, double brightness) {
        TextView myTextView = findViewById(R.id.user_Message);
        TextView Image_bright = findViewById(R.id.Image_bright);

        myTextView.setVisibility(View.GONE);
        myTextView.setText("");
        Image_bright.setVisibility(View.GONE);
        Image_bright.setText("");

        boolean isBadQuality = false;

        // Check focus & clarity
        if (edgeDensity > 0.06 && sharpness > 300) {
            myTextView.append("* The image is sharp and well-focused with great edge detail.\n");

            if (contrast > 249) {
                myTextView.append("* Excellent contrast detected.\n");
            } else if (contrast >= 200 && contrast <= 249) {
                myTextView.append("* Contrast is acceptable.\n");
            } else if (contrast < 150) {
                isBadQuality = true;
                myTextView.append("* Image has very low contrast.\n");
            } else {
                isBadQuality = true;
                myTextView.append("* Image contrast is slightly low.\n");
            }
        } else if (edgeDensity == 0.07 && contrast < 249) {
            isBadQuality = true;
            myTextView.append("* The image lacks proper focus.\n");
        } else if (edgeDensity < 0.07 && sharpness < 300) {
            isBadQuality = true;
            myTextView.append("* The image lacks proper focus.\n");
        } else {
            isBadQuality = true;
            myTextView.append("* The image seems unclear and not focused properly.\n");
        }

        // Check brightness
        if (brightness >= 0 && brightness <= 50) {
            isBadQuality = true;
            myTextView.append("* Image is extremely underexposed.\n");
        } else if (brightness > 50 && brightness <= 80) {
            isBadQuality = true;
            myTextView.append("* Image is too dark.\n");
        } else if (brightness > 80 && brightness <= 120) {
//            isBadQuality = true;
            myTextView.append("* Image could use a bit more lighting.\n");
        } else if (brightness > 120 && brightness <= 170) {
            // Ideal range
        } else if (brightness > 170 && brightness <= 230) {
            isBadQuality = true;
            myTextView.append("* Image is overexposed in light.\n");
        } else if (brightness > 230 && brightness <= 255) {
            isBadQuality = true;
            myTextView.append("* Image is highly overexposed in light.\n");
        }

        // Final message
        String qualityLevel = isBadQuality ? "Bad" : "Good";
        String finalMessage = "Image Quality: " + qualityLevel;
        Image_bright.setText(finalMessage);
        Image_bright.setVisibility(View.VISIBLE);
        String existingText = myTextView.getText().toString();
        String newText = "Reasons : \n";
        myTextView.setText(newText + existingText);
        myTextView.setVisibility(View.VISIBLE);

    }

    public void fixAutoBrightness(String imagePath) throws IOException {
        File tempFile  = createTempFileFromUri(Uri.parse(imagePath));
        Bitmap bitmap = BitmapFactory.decodeFile(tempFile.getAbsolutePath());
        Bitmap resizedBitmap = getResizedBitmapCV(bitmap, 300, 300);
        Log.d("Image path in the fix auto brightness",imagePath);
        // Load input image
        Mat image = Imgcodecs.imread(tempFile.getAbsolutePath());
        Log.d("Log in second line", String.valueOf(image));
        if (image.empty()) {
            System.out.println("Image not found.");
            return;
        }

        // Convert to HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(image, hsv, Imgproc.COLOR_BGR2HSV);

        // Split channels
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);

        Mat vChannel = hsvChannels.get(2); // V channel = brightness

        // Calculate average brightness
        Scalar avgBrightnessScalar = Core.mean(vChannel);
        double avgBrightness = avgBrightnessScalar.val[0];
        System.out.println("Average Brightness: " + avgBrightness);

        // Set thresholds and adjustment strength
        double underexposedThreshold = 60;
        double overexposedThreshold = 190;
        double adjustmentStrength = 70;

        // Adjust V channel based on brightness
        if (avgBrightness < underexposedThreshold) {
            System.out.println("Underexposed. Increasing brightness...");
            Core.add(vChannel, new Scalar(adjustmentStrength), vChannel);
        } else if (avgBrightness > overexposedThreshold) {
            System.out.println("Overexposed. Reducing brightness...");
            Core.subtract(vChannel, new Scalar(adjustmentStrength), vChannel);
        } else {
            System.out.println("Exposure is okay. No change needed.");
        }

        // Clip values to [0, 255]
        Core.min(vChannel, new Scalar(255), vChannel);
        Core.max(vChannel, new Scalar(0), vChannel);

        // Merge V back and convert to BGR
        hsvChannels.set(2, vChannel);
        Core.merge(hsvChannels, hsv);

        Mat correctedImage = new Mat();
        Imgproc.cvtColor(hsv, correctedImage, Imgproc.COLOR_HSV2BGR);

        // Save corrected image
        Imgcodecs.imwrite("path/to/output/corrected_image.jpg", correctedImage);
        System.out.println("Exposure correction completed.");
        Bitmap outputBitmap = Bitmap.createBitmap(correctedImage.cols(), correctedImage.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(correctedImage, outputBitmap);

// Set to ImageView (assumes you're calling from an Activity or have reference)
//        ImageView imageView = findViewById(R.id.imageView); // or pass ImageView as param
//        imageView.setImageBitmap(outputBitmap);
    }
    private void requestPermissions() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this,
                    new String[]{android.Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE},
                    REQUEST_PERMISSIONS);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                analyzeImage(String.valueOf(photoUri));
                selectedImageView.setImageURI(photoUri);
                selectedImageView.setVisibility(View.VISIBLE);
            } else if (requestCode == REQUEST_GALLERY_PICK && data != null) {
                Log.d("data", String.valueOf(data));
                Uri selectedImageUri = data.getData();
                Log.d("selectedImageUri", String.valueOf(selectedImageUri));
                analyzeImage(String.valueOf(selectedImageUri));
                selectedImageView.setImageURI(selectedImageUri);
                selectedImageView.setVisibility(View.VISIBLE);
                try {
                    fixAutoBrightness(String.valueOf(selectedImageUri));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }else if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
                // photoUri contains the image URI
                analyzeImage(String.valueOf(photoUri));
                selectedImageView.setImageURI(photoUri);
                selectedImageView.setVisibility(View.VISIBLE);
                try {
                    fixAutoBrightness(String.valueOf(photoUri));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                Toast.makeText(this, "Image saved: " + photoUri.getPath(), Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void pickImageFromGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(pickPhoto, REQUEST_GALLERY_PICK);
    }


}