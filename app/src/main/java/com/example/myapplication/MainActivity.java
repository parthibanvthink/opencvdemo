package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.core.view.WindowCompat;
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
import org.opencv.imgproc.Imgproc;
import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import android.Manifest;
import android.widget.Button;
import android.widget.ImageView;
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

    @Override
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
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
//        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
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
            isBadQuality = true;
            myTextView.append("* Image could use a bit more lighting.\n");
        } else if (brightness > 120 && brightness <= 170) {
            // Ideal range
        } else if (brightness > 170 && brightness <= 230) {
            isBadQuality = true;
            myTextView.append("* Image is overexposed.\n");
        } else if (brightness > 230 && brightness <= 255) {
            isBadQuality = true;
            myTextView.append("* Image is highly overexposed.\n");
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
                Uri selectedImageUri = data.getData();
                analyzeImage(String.valueOf(selectedImageUri));
                selectedImageView.setImageURI(selectedImageUri);
                selectedImageView.setVisibility(View.VISIBLE);
            }else if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
                // photoUri contains the image URI
                analyzeImage(String.valueOf(photoUri));
                selectedImageView.setImageURI(photoUri);
                selectedImageView.setVisibility(View.VISIBLE);
                Toast.makeText(this, "Image saved: " + photoUri.getPath(), Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void pickImageFromGallery() {
        Intent pickPhoto = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        Log.d("Pick photo data", String.valueOf(pickPhoto));
        startActivityForResult(pickPhoto, REQUEST_GALLERY_PICK);
    }


}