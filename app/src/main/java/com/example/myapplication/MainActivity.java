package com.example.myapplication;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.util.Log;
import android.view.View;

import androidx.core.view.WindowCompat;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.example.myapplication.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity {
    private static final String IMAGE_FILE_PATH = "/storage/emulated/0/DCIM/Camera/20250418_155340.jpg";

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);

        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAnchorView(R.id.fab)
                        .setAction("Action", null).show();
            }
        });
        if (OpenCVLoader.initLocal()) {
            Log.i("OpenCV", "OpenCV successfully loaded.");
        }
        analyzeImage(IMAGE_FILE_PATH);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
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
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
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
            String result = analyzeImage(brightness,sharpness);
            Log.d("ImageAnalysis result",result);
            // Build JSON output
            String jsonOutput = String.format("{\"brightness\": %.2f, \"sharpness\": %.2f,\"edgeDensity\": %.2f,\"contrast\": %.2f}", brightness, sharpness,edgeDensity,contrast);
            Log.d("ImageAnalysis", jsonOutput); // print the result to Logcat

            // (Optional) You can return this string if calling from another method
            // return jsonOutput;

        } catch (Exception e) {
            Log.e("ImageAnalysis", "Error analyzing image", e);
        }
    }
    public String analyzeImage(Double brightness, Double sharpness) {
        // Brightness thresholds (0-255 scale for grayscale)
        if (brightness < 40) {
            return "Too dark";
        }

        if (brightness > 220) {
            return "Too bright";
        }

        // Sharpness threshold (variance of Laplacian): <100 is considered blurry
        if (sharpness < 400) {
            return "Blurry";
        }

        return "Good";
    }
}