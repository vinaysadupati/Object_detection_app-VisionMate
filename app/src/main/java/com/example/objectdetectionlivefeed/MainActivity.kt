package com.example.objectdetectionlivefeed

import android.Manifest
import android.app.Fragment
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraManager
import android.media.Image.Plane
import android.media.ImageReader
import android.media.MediaPlayer
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.Surface
import android.view.View
import android.widget.TextView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.imageclassificationlivefeed.CameraConnectionFragment
import com.example.imageclassificationlivefeed.ImageUtils.convertYUV420ToARGB8888
import com.example.imageclassificationlivefeed.ImageUtils.getTransformationMatrix
import com.example.objectdetectionlivefeed.Drawing.BorderedText
import com.example.objectdetectionlivefeed.Drawing.MultiBoxTracker
import com.example.objectdetectionlivefeed.Drawing.OverlayView
import org.tensorflow.lite.examples.detection.tflite.Detector
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel
import java.io.IOException
import java.util.*
import android.os.Handler
import android.os.Looper
import kotlin.math.roundToInt


class MainActivity : AppCompatActivity(), ImageReader.OnImageAvailableListener {
    private var detectionCounts = mutableMapOf<String, Int>()
    private val maxAnnouncements = 2
    private var handler = Handler(Looper.getMainLooper())
    private var resultTV: TextView? = null
    private var detector: Detector? = null
    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null
    private var minimumConfidence: Float = 0.5f
    private val MAINTAIN_ASPECT = false
    private val TEXT_SIZE_DIP = 10f
    private var trackingOverlay: OverlayView? = null
    private var borderedText: BorderedText? = null
    private lateinit var textToSpeech: TextToSpeech
    private lateinit var mediaPlayer: MediaPlayer
    private var previousDistances = mutableMapOf<String, Float>()
    private val horizontalFoV = 60.0 
    private val verticalFoV = 45.0 
    private var focalLengthPixels = 0.0f

    // Parameters for distance calculation
    val knownObjectHeight = mapOf(
        // Vehicles
        "person" to 1.0f,
        "bicycle" to 1.0f,
        "car" to 1.5f,
        "motorcycle" to 1.2f,
        "airplane" to 5.0f, // Depending on the model, this can vary greatly
        "bus" to 3.2f,
        "train" to 4.0f,
        "truck" to 3.0f,
        "boat" to 2.5f,

        // Road Objects
        "traffic light" to 2.0f,
        "fire hydrant" to 0.75f,
        "stop sign" to 2.0f,
        "parking meter" to 1.2f,

        // Furniture
        "bench" to 0.6f,
        "chair" to 0.9f,
        "couch" to 0.8f,
        "potted plant" to 0.5f,
        "bed" to 0.5f,
        "dining table" to 0.75f,

        // Animals
        "bird" to 0.2f,
        "cat" to 0.3f,
        "dog" to 0.5f,
        "horse" to 1.6f,
        "sheep" to 1.0f,
        "cow" to 1.5f,
        "elephant" to 3.0f,
        "bear" to 2.0f,
        "zebra" to 1.4f,
        "giraffe" to 4.5f,

        // Personal Items
        "backpack" to 0.5f,
        "umbrella" to 1.0f,
        "handbag" to 0.3f,
        "tie" to 0.6f,
        "suitcase" to 0.8f,

        // Sports Items
        "frisbee" to 0.25f,
        "skis" to 1.7f,
        "snowboard" to 1.5f,
        "sports ball" to 0.25f,
        "kite" to 0.5f,
        "baseball bat" to 1.0f,
        "baseball glove" to 0.25f,
        "skateboard" to 0.8f,
        "surfboard" to 2.0f,
        "tennis racket" to 0.7f,

        // Kitchenware
        "bottle" to 0.25f,
        "wine glass" to 0.2f,
        "cup" to 0.15f,
        "fork" to 0.2f,
        "knife" to 0.2f,
        "spoon" to 0.2f,
        "bowl" to 0.1f,

        // Food
        "banana" to 0.2f,
        "apple" to 0.1f,
        "sandwich" to 0.05f,
        "orange" to 0.1f,
        "broccoli" to 0.25f,
        "carrot" to 0.2f,
        "hot dog" to 0.2f,
        "pizza" to 0.3f,
        "donut" to 0.1f,
        "cake" to 0.15f,

        // Appliances
        "microwave" to 0.4f,
        "oven" to 0.9f,
        "toaster" to 0.3f,
        "sink" to 0.5f,
        "refrigerator" to 1.8f,
        "tv" to 0.6f,
        "laptop" to 0.3f,
        "mouse" to 0.05f,
        "remote" to 0.2f,
        "keyboard" to 0.15f,
        "cell phone" to 0.15f,

        // Miscellaneous
        "book" to 0.3f,
        "clock" to 0.3f,
        "vase" to 0.5f,
        "scissors" to 0.2f,
        "teddy bear" to 0.4f,
        "hair dryer" to 0.3f,
        "toothbrush" to 0.2f
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize the MediaPlayer with the buzzer sound
        mediaPlayer = MediaPlayer.create(this, R.raw.buzzer1)

        // Handle permissions for camera usage
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                val permission = arrayOf(Manifest.permission.CAMERA)
                requestPermissions(permission, 1122)
            } else {
                setFragment()
            }
        } else {
            setFragment()
        }

        resultTV = findViewById(R.id.textView)

        // Initialize TextToSpeech for audio output
        textToSpeech = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.language = Locale.US
            } else {
                Log.e("TTS", "Initialization failed")
            }
        }



        // Initialize the object detector
        try {
            detector = TFLiteObjectDetectionAPIModel.create(
                this,
                "efficientdet_lite3.tflite",
                "labelmap.txt",
                320,
                true
            )
            Log.d("tryLog", "Detector initialized successfully")
        } catch (e: IOException) {
            Log.d("tryException", "Error initializing detector: ${e.message}")
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String?>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            setFragment()
        } else {
            finish()
        }
    }

    private var previewHeight = 0
    private var previewWidth = 0
    private var sensorOrientation = 0

    protected fun setFragment() {
        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        var cameraId: String? = null
        try {
            cameraId = manager.cameraIdList[0]
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
        val fragment: Fragment
        val camera2Fragment = CameraConnectionFragment.newInstance(
            object : CameraConnectionFragment.ConnectionCallback {
                override fun onPreviewSizeChosen(size: Size?, rotation: Int) {
                    previewHeight = size!!.height
                    previewWidth = size.width
                    val textSizePx = TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics
                    )
                    borderedText = BorderedText(textSizePx)
                    borderedText!!.setTypeface(Typeface.MONOSPACE)
                    tracker = MultiBoxTracker(this@MainActivity)

                    val cropSize = 300
                    previewWidth = size.width
                    previewHeight = size.height
                    sensorOrientation = rotation - getScreenOrientation()
                    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888)

                    frameToCropTransform = getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT
                    )
                    cropToFrameTransform = Matrix()
                    frameToCropTransform!!.invert(cropToFrameTransform)

                    // Calculate focal length in pixels based on camera FoV (assuming 60 degrees here)
                    focalLengthPixels = (previewWidth / (2 * Math.tan(Math.toRadians(60.0) / 2))).toFloat()

                    trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
                    trackingOverlay!!.addCallback(
                        object : OverlayView.DrawCallback {
                            override fun drawCallback(canvas: Canvas?) {
                                tracker!!.draw(canvas!!)
                            }
                        })
                    tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation)
                }
            },
            this, R.layout.camera_fragment, Size(640, 480)
        )
        camera2Fragment.setCamera(cameraId)
        fragment = camera2Fragment
        fragmentManager.beginTransaction().replace(R.id.container, fragment).commit()
    }

    private var isProcessingFrame = false
    private val yuvBytes = arrayOfNulls<ByteArray>(3)
    private var rgbBytes: IntArray? = null
    private var yRowStride = 0
    private var postInferenceCallback: Runnable? = null
    private var imageConverter: Runnable? = null
    private var rgbFrameBitmap: Bitmap? = null

    protected fun fillBytes(planes: Array<Plane>, yuvBytes: Array<ByteArray?>) {
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer.get(yuvBytes[i])
        }
    }

    @RequiresApi(Build.VERSION_CODES.N)
    override fun onImageAvailable(reader: ImageReader) {
        if (previewWidth == 0 || previewHeight == 0) return
        if (rgbBytes == null) {
            rgbBytes = IntArray(previewWidth * previewHeight)
        }
        try {
            val image = reader.acquireLatestImage() ?: return
            if (isProcessingFrame) {
                image.close()
                return
            }
            isProcessingFrame = true
            val planes = image.planes
            fillBytes(planes, yuvBytes)
            yRowStride = planes[0].rowStride
            val uvRowStride = planes[1].rowStride
            val uvPixelStride = planes[1].pixelStride
            imageConverter = Runnable {
                convertYUV420ToARGB8888(
                    yuvBytes[0]!!, yuvBytes[1]!!, yuvBytes[2]!!,
                    previewWidth, previewHeight, yRowStride, uvRowStride, uvPixelStride, rgbBytes!!
                )
            }
            postInferenceCallback = Runnable {
                image.close()
                isProcessingFrame = false
            }
            processImage()
        } catch (e: Exception) {
            Log.d("tryError", "Image processing error: ${e.message}")
            return
        }
    }

    private var croppedBitmap: Bitmap? = null
    private var tracker: MultiBoxTracker? = null

    @RequiresApi(Build.VERSION_CODES.N)
    fun processImage() {
        imageConverter!!.run()
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)
        rgbFrameBitmap!!.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight)

        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)

        val results = detector!!.recognizeImage(croppedBitmap)
        results.removeIf { it.confidence < minimumConfidence }
        for (result in results) {
            val location: RectF = result.location
            cropToFrameTransform!!.mapRect(location)
            result.location = location
        }

        tracker?.trackResults(results, 10)
        trackingOverlay?.postInvalidate()
        postInferenceCallback!!.run()

        val objectsToAnnounce = mutableListOf<String>()
        var objectNear = false // Flag to indicate if any object is close

        for (result in results) {
            val detectedObject = result.title
            val location: RectF = result.location

            val objectHeightInImage = location.height() // Height of the object in pixels
            val objectHeight = knownObjectHeight[detectedObject] ?: continue // Skip if height is not found

            // Calculate the distance
            val distance = (objectHeight * focalLengthPixels / objectHeightInImage)

            if (distance < 0.75f && !mediaPlayer.isPlaying) {
                mediaPlayer.start() // Play the buzzer sound
            }

            // Check if we have a previous distance to compare with
            val movementStatus = when {
                previousDistances[detectedObject]?.let { it > distance } == true -> "moving towards you"
                previousDistances[detectedObject]?.let { it < distance } == true -> "moving away from you"
                else -> "not moving anywhere"
            }

            // Update the previous distance for the detected object
            previousDistances[detectedObject] = distance

            // Update this section in processImage to calculate angles with improved accuracy
            val objectCenterX = location.centerX()
            val objectCenterY = location.centerY()
            val frameCenterX = previewWidth / 2f
            val frameCenterY = previewHeight / 2f

            // Calculate the relative position of the object center in the frame
            val deltaX = objectCenterX - frameCenterX
            val deltaY = objectCenterY - frameCenterY

            // Calculate the angle using the relative position scaled by the FoV, then normalize
            val rawAngleX = Math.toDegrees(deltaX / frameCenterX * (horizontalFoV / 2))
            val rawAngleY = Math.toDegrees(deltaY / frameCenterY * (verticalFoV / 2))

            // Normalize angles to be within -180° to +180° range
            val angleX = rawAngleX.coerceIn(-horizontalFoV / 2, horizontalFoV / 2).roundToInt()
            val angleY = rawAngleY.coerceIn(-verticalFoV / 2, verticalFoV / 2).roundToInt()

            // Add information to the announcements
            objectsToAnnounce.add(
                "There is a $detectedObject approximately ${"%.2f".format(distance)} meters away at angle (${angleX}°, ${angleY}°), $movementStatus."
            )
        }

        // Speak out the announcements
        if (objectsToAnnounce.isNotEmpty() && !textToSpeech.isSpeaking) {
            val speechText = objectsToAnnounce.joinToString(", ")
            textToSpeech.speak(speechText, TextToSpeech.QUEUE_FLUSH, null, null)
        }

        // Clear detection counts after a delay to avoid repeated announcements
        handler.postDelayed({
            for (detectedObject in objectsToAnnounce) {
                detectionCounts.remove(detectedObject)
            }
        }, 5000)
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        mediaPlayer.release()
        textToSpeech.stop()
        textToSpeech.shutdown()
    }

    private fun getScreenOrientation(): Int {
        return when (windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_270 -> 270
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_90 -> 90
            else -> 0
        }
    }
}
