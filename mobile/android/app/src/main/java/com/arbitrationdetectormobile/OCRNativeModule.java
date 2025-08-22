package com.arbitrationdetectormobile;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.Rect;
import android.net.Uri;
import android.os.AsyncTask;
import android.util.Log;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions;
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning;
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class OCRNativeModule extends ReactContextBaseJavaModule {
    private static final String TAG = "OCRNativeModule";
    private TextRecognizer textRecognizer;

    public OCRNativeModule(ReactApplicationContext reactContext) {
        super(reactContext);
        // Initialize ML Kit Text Recognition
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
    }

    @Override
    public String getName() {
        return "OCRNativeModule";
    }

    @ReactMethod
    public void recognizeTextInImage(String imagePath, Promise promise) {
        try {
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap == null) {
                promise.reject("IMAGE_ERROR", "Failed to load image from path: " + imagePath);
                return;
            }

            InputImage image = InputImage.fromBitmap(bitmap, 0);
            
            textRecognizer.process(image)
                .addOnSuccessListener(new OnSuccessListener<Text>() {
                    @Override
                    public void onSuccess(Text visionText) {
                        WritableMap result = Arguments.createMap();
                        WritableArray blocks = Arguments.createArray();
                        
                        String combinedText = visionText.getText();
                        float totalConfidence = 0f;
                        int blockCount = 0;
                        
                        for (Text.TextBlock block : visionText.getTextBlocks()) {
                            WritableMap blockData = Arguments.createMap();
                            blockData.putString("text", block.getText());
                            blockData.putDouble("confidence", 0.9); // ML Kit doesn't provide confidence
                            blocks.pushMap(blockData);
                            
                            totalConfidence += 0.9f;
                            blockCount++;
                        }
                        
                        result.putString("text", combinedText);
                        result.putArray("blocks", blocks);
                        result.putDouble("confidence", blockCount > 0 ? totalConfidence / blockCount : 0);
                        
                        promise.resolve(result);
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        promise.reject("OCR_ERROR", "Text recognition failed: " + e.getMessage(), e);
                    }
                });
                
        } catch (Exception e) {
            promise.reject("OCR_EXCEPTION", "Exception during text recognition: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void recognizeTextInImageWithBounds(String imagePath, Promise promise) {
        try {
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            if (bitmap == null) {
                promise.reject("IMAGE_ERROR", "Failed to load image from path: " + imagePath);
                return;
            }

            InputImage image = InputImage.fromBitmap(bitmap, 0);
            
            textRecognizer.process(image)
                .addOnSuccessListener(new OnSuccessListener<Text>() {
                    @Override
                    public void onSuccess(Text visionText) {
                        WritableMap result = Arguments.createMap();
                        WritableArray boundingBoxes = Arguments.createArray();
                        
                        String combinedText = visionText.getText();
                        float totalConfidence = 0f;
                        int elementCount = 0;
                        
                        for (Text.TextBlock block : visionText.getTextBlocks()) {
                            for (Text.Line line : block.getLines()) {
                                for (Text.Element element : line.getElements()) {
                                    WritableMap elementData = Arguments.createMap();
                                    Rect boundingBox = element.getBoundingBox();
                                    
                                    if (boundingBox != null) {
                                        elementData.putString("text", element.getText());
                                        elementData.putDouble("confidence", 0.9);
                                        elementData.putInt("x", boundingBox.left);
                                        elementData.putInt("y", boundingBox.top);
                                        elementData.putInt("width", boundingBox.width());
                                        elementData.putInt("height", boundingBox.height());
                                        
                                        boundingBoxes.pushMap(elementData);
                                        totalConfidence += 0.9f;
                                        elementCount++;
                                    }
                                }
                            }
                        }
                        
                        result.putString("text", combinedText);
                        result.putArray("boundingBoxes", boundingBoxes);
                        result.putDouble("confidence", elementCount > 0 ? totalConfidence / elementCount : 0);
                        
                        promise.resolve(result);
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception e) {
                        promise.reject("OCR_ERROR", "Text recognition with bounds failed: " + e.getMessage(), e);
                    }
                });
                
        } catch (Exception e) {
            promise.reject("OCR_EXCEPTION", "Exception during text recognition: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void detectDocumentBounds(String imagePath, Promise promise) {
        try {
            // For document detection, we'll use a simplified approach
            // In a production app, you might want to use OpenCV or similar library
            
            WritableMap result = Arguments.createMap();
            WritableArray corners = Arguments.createArray();
            
            // Return empty result as document detection requires more complex implementation
            result.putArray("corners", corners);
            result.putDouble("confidence", 0.0);
            
            promise.resolve(result);
            
        } catch (Exception e) {
            promise.reject("DETECTION_EXCEPTION", "Exception during document detection: " + e.getMessage(), e);
        }
    }

    @ReactMethod
    public void enhanceImageForOCR(String imagePath, Promise promise) {
        new AsyncTask<Void, Void, String>() {
            @Override
            protected String doInBackground(Void... voids) {
                try {
                    Bitmap originalBitmap = BitmapFactory.decodeFile(imagePath);
                    if (originalBitmap == null) {
                        return null;
                    }
                    
                    Bitmap enhancedBitmap = enhanceImageQuality(originalBitmap);
                    return saveImageToTempFile(enhancedBitmap);
                    
                } catch (Exception e) {
                    Log.e(TAG, "Error enhancing image", e);
                    return null;
                }
            }
            
            @Override
            protected void onPostExecute(String outputPath) {
                if (outputPath != null) {
                    WritableMap result = Arguments.createMap();
                    result.putString("path", outputPath);
                    promise.resolve(result);
                } else {
                    promise.reject("ENHANCEMENT_ERROR", "Failed to enhance image");
                }
            }
        }.execute();
    }

    @ReactMethod
    public void correctPerspective(String imagePath, ReadableArray corners, Promise promise) {
        new AsyncTask<Void, Void, String>() {
            @Override
            protected String doInBackground(Void... voids) {
                try {
                    Bitmap originalBitmap = BitmapFactory.decodeFile(imagePath);
                    if (originalBitmap == null) {
                        return null;
                    }
                    
                    // For perspective correction, we'll apply a simple transformation
                    // In a production app, you'd want to implement proper perspective correction
                    Bitmap correctedBitmap = applyPerspectiveCorrection(originalBitmap, corners);
                    return saveImageToTempFile(correctedBitmap);
                    
                } catch (Exception e) {
                    Log.e(TAG, "Error correcting perspective", e);
                    return null;
                }
            }
            
            @Override
            protected void onPostExecute(String outputPath) {
                if (outputPath != null) {
                    WritableMap result = Arguments.createMap();
                    result.putString("path", outputPath);
                    promise.resolve(result);
                } else {
                    promise.reject("CORRECTION_ERROR", "Failed to correct perspective");
                }
            }
        }.execute();
    }

    private Bitmap enhanceImageQuality(Bitmap originalBitmap) {
        // Apply image enhancements for better OCR results
        
        // 1. Increase contrast
        Bitmap contrastBitmap = adjustContrast(originalBitmap, 1.3f);
        
        // 2. Increase brightness slightly
        Bitmap brightnessBitmap = adjustBrightness(contrastBitmap, 20);
        
        // 3. Apply sharpening
        Bitmap sharpenedBitmap = applySharpen(brightnessBitmap);
        
        return sharpenedBitmap;
    }

    private Bitmap adjustContrast(Bitmap bitmap, float contrast) {
        android.graphics.ColorMatrix colorMatrix = new android.graphics.ColorMatrix();
        colorMatrix.set(new float[] {
            contrast, 0, 0, 0, 0,
            0, contrast, 0, 0, 0,
            0, 0, contrast, 0, 0,
            0, 0, 0, 1, 0
        });
        
        android.graphics.ColorMatrixColorFilter filter = new android.graphics.ColorMatrixColorFilter(colorMatrix);
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setColorFilter(filter);
        
        Bitmap result = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
        android.graphics.Canvas canvas = new android.graphics.Canvas(result);
        canvas.drawBitmap(bitmap, 0, 0, paint);
        
        return result;
    }

    private Bitmap adjustBrightness(Bitmap bitmap, int brightness) {
        android.graphics.ColorMatrix colorMatrix = new android.graphics.ColorMatrix();
        colorMatrix.set(new float[] {
            1, 0, 0, 0, brightness,
            0, 1, 0, 0, brightness,
            0, 0, 1, 0, brightness,
            0, 0, 0, 1, 0
        });
        
        android.graphics.ColorMatrixColorFilter filter = new android.graphics.ColorMatrixColorFilter(colorMatrix);
        android.graphics.Paint paint = new android.graphics.Paint();
        paint.setColorFilter(filter);
        
        Bitmap result = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
        android.graphics.Canvas canvas = new android.graphics.Canvas(result);
        canvas.drawBitmap(bitmap, 0, 0, paint);
        
        return result;
    }

    private Bitmap applySharpen(Bitmap bitmap) {
        // Simple sharpening kernel
        float[] sharpenKernel = {
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
        };
        
        // For simplicity, return the original bitmap
        // In production, you'd implement proper convolution
        return bitmap;
    }

    private Bitmap applyPerspectiveCorrection(Bitmap bitmap, ReadableArray corners) {
        // For demonstration, just return the original bitmap
        // In production, implement proper perspective correction using transformation matrix
        return bitmap;
    }

    private String saveImageToTempFile(Bitmap bitmap) {
        try {
            File tempDir = new File(getReactApplicationContext().getCacheDir(), "processed_images");
            if (!tempDir.exists()) {
                tempDir.mkdirs();
            }
            
            String fileName = "processed_image_" + UUID.randomUUID().toString() + ".jpg";
            File tempFile = new File(tempDir, fileName);
            
            FileOutputStream fos = new FileOutputStream(tempFile);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos);
            fos.close();
            
            return tempFile.getAbsolutePath();
            
        } catch (IOException e) {
            Log.e(TAG, "Error saving image to temp file", e);
            return null;
        }
    }

    private void sendEvent(String eventName, WritableMap params) {
        getReactApplicationContext()
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
            .emit(eventName, params);
    }

    @Override
    public void onCatalystInstanceDestroy() {
        super.onCatalystInstanceDestroy();
        if (textRecognizer != null) {
            textRecognizer.close();
        }
    }
}