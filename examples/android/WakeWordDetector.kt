package com.wakeword.detector

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Base64
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * WakeWord Detector for Android
 *
 * Detects custom wake words using either:
 * 1. Server-based inference (sends audio to WakeWord server)
 * 2. Local inference using ONNX Runtime (offline capable)
 *
 * Usage:
 * ```kotlin
 * val detector = WakeWordDetector(context, "http://server:8000", "jarvis")
 * detector.initialize()
 * detector.startListening { confidence ->
 *     if (confidence > 0.5) {
 *         Log.d("WakeWord", "Detected!")
 *     }
 * }
 * ```
 */

// ============================================================================
// Data Classes
// ============================================================================

data class WakeWordConfig(
    val word: String,
    val serverUrl: String = "http://localhost:8000",
    val threshold: Float = 0.5f,
    val sampleRate: Int = 16000,
    val chunkDurationMs: Int = 1500,
)

data class TrainingStatus(
    val jobId: String,
    val word: String,
    val status: String,
    val progress: Float,
    val currentStage: String,
    val errorMessage: String? = null,
) {
    val isComplete: Boolean get() = status == "completed"
    val isFailed: Boolean get() = status == "failed"
    val isInProgress: Boolean get() = !isComplete && !isFailed && status != "cancelled"
}

data class PredictionResult(
    val detected: Boolean,
    val confidence: Float,
    val word: String,
)

sealed class DetectorState {
    object Uninitialized : DetectorState()
    object Initializing : DetectorState()
    object Ready : DetectorState()
    object Listening : DetectorState()
    data class Error(val message: String) : DetectorState()
}

// ============================================================================
// WakeWord Client
// ============================================================================

class WakeWordClient(private val serverUrl: String) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .readTimeout(30, java.util.concurrent.TimeUnit.SECONDS)
        .build()

    companion object {
        private const val TAG = "WakeWordClient"
    }

    /**
     * Check if server is healthy
     */
    suspend fun isHealthy(): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$serverUrl/health")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            response.isSuccessful
        } catch (e: Exception) {
            Log.e(TAG, "Health check failed: ${e.message}")
            false
        }
    }

    /**
     * Check if model exists for a word
     */
    suspend fun modelExists(word: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$serverUrl/models/$word/config")
                .get()
                .build()
            
            val response = client.newCall(request).execute()
            response.isSuccessful
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Request training for a word
     */
    suspend fun requestTraining(
        word: String,
        sampleCount: Int = 500,
        epochs: Int = 50,
    ): TrainingStatus = withContext(Dispatchers.IO) {
        val json = JSONObject().apply {
            put("word", word)
            put("sample_count", sampleCount)
            put("epochs", epochs)
        }

        val request = Request.Builder()
            .url("$serverUrl/train")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Training request failed: ${response.body?.string()}")
        }

        val body = JSONObject(response.body?.string() ?: "{}")
        TrainingStatus(
            jobId = body.optString("job_id", ""),
            word = body.optString("word", word),
            status = body.optString("status", "pending"),
            progress = body.optDouble("progress", 0.0).toFloat(),
            currentStage = body.optString("current_stage", ""),
        )
    }

    /**
     * Get training job status
     */
    suspend fun getJobStatus(jobId: String): TrainingStatus = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$serverUrl/jobs/$jobId")
            .get()
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Failed to get job status: ${response.body?.string()}")
        }

        val body = JSONObject(response.body?.string() ?: "{}")
        TrainingStatus(
            jobId = body.optString("job_id", jobId),
            word = body.optString("word", ""),
            status = body.optString("status", ""),
            progress = body.optDouble("progress", 0.0).toFloat(),
            currentStage = body.optString("current_stage", ""),
            errorMessage = body.optString("error_message", null),
        )
    }

    /**
     * Wait for training to complete
     */
    suspend fun waitForTraining(
        jobId: String,
        pollIntervalMs: Long = 10000,
        onProgress: ((TrainingStatus) -> Unit)? = null,
    ): TrainingStatus {
        while (true) {
            val status = getJobStatus(jobId)
            onProgress?.invoke(status)

            when {
                status.isComplete -> return status
                status.isFailed -> throw IOException("Training failed: ${status.errorMessage}")
            }

            delay(pollIntervalMs)
        }
    }

    /**
     * Download model bytes
     */
    suspend fun downloadModel(word: String, format: String = "onnx"): ByteArray = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("$serverUrl/models/$word?format=$format")
            .get()
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Failed to download model: ${response.body?.string()}")
        }

        response.body?.bytes() ?: throw IOException("Empty response")
    }

    /**
     * Predict wake word from audio
     */
    suspend fun predict(word: String, audioBytes: ByteArray): PredictionResult = withContext(Dispatchers.IO) {
        val audioBase64 = Base64.encodeToString(audioBytes, Base64.NO_WRAP)
        
        val json = JSONObject().apply {
            put("audio_base64", audioBase64)
        }

        val request = Request.Builder()
            .url("$serverUrl/predict/$word")
            .post(json.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            throw IOException("Prediction failed: ${response.body?.string()}")
        }

        val body = JSONObject(response.body?.string() ?: "{}")
        PredictionResult(
            detected = body.optBoolean("detected", false),
            confidence = body.optDouble("confidence", 0.0).toFloat(),
            word = body.optString("word", word),
        )
    }

    /**
     * Ensure model exists, training if necessary
     */
    suspend fun ensureModel(
        word: String,
        sampleCount: Int = 500,
        epochs: Int = 50,
        onProgress: ((TrainingStatus) -> Unit)? = null,
    ) {
        if (modelExists(word)) {
            Log.d(TAG, "Model for '$word' already exists")
            return
        }

        Log.d(TAG, "Requesting training for '$word'...")
        val job = requestTraining(word, sampleCount, epochs)
        
        if (!job.isComplete) {
            waitForTraining(job.jobId, onProgress = onProgress)
        }
    }
}

// ============================================================================
// Audio Recorder
// ============================================================================

class AudioRecorder(
    private val sampleRate: Int = 16000,
    private val chunkDurationMs: Int = 1500,
) {
    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private var recordingJob: Job? = null

    val bufferSize: Int
        get() = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(sampleRate * chunkDurationMs / 1000 * 2)

    fun start(context: Context, onAudioChunk: (ByteArray) -> Unit): Boolean {
        if (ActivityCompat.checkSelfPermission(
                context, Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            return false
        }

        val chunkSamples = sampleRate * chunkDurationMs / 1000
        val chunkBytes = chunkSamples * 2  // 16-bit = 2 bytes

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )

        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            return false
        }

        isRecording = true
        audioRecord?.startRecording()

        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            val buffer = ShortArray(chunkSamples)
            
            while (isRecording) {
                val read = audioRecord?.read(buffer, 0, chunkSamples) ?: 0
                if (read > 0) {
                    // Convert to WAV bytes
                    val wavBytes = pcmToWav(buffer, read, sampleRate)
                    onAudioChunk(wavBytes)
                }
            }
        }

        return true
    }

    fun stop() {
        isRecording = false
        recordingJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun pcmToWav(pcmData: ShortArray, numSamples: Int, sampleRate: Int): ByteArray {
        val byteRate = sampleRate * 2  // 16-bit mono
        val dataSize = numSamples * 2
        val totalSize = 44 + dataSize

        val buffer = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)

        // RIFF header
        buffer.put("RIFF".toByteArray())
        buffer.putInt(totalSize - 8)
        buffer.put("WAVE".toByteArray())

        // fmt chunk
        buffer.put("fmt ".toByteArray())
        buffer.putInt(16)  // chunk size
        buffer.putShort(1)  // PCM format
        buffer.putShort(1)  // mono
        buffer.putInt(sampleRate)
        buffer.putInt(byteRate)
        buffer.putShort(2)  // block align
        buffer.putShort(16)  // bits per sample

        // data chunk
        buffer.put("data".toByteArray())
        buffer.putInt(dataSize)
        
        for (i in 0 until numSamples) {
            buffer.putShort(pcmData[i])
        }

        return buffer.array()
    }
}

// ============================================================================
// Wake Word Detector
// ============================================================================

class WakeWordDetector(
    private val context: Context,
    private val config: WakeWordConfig,
) {
    private val client = WakeWordClient(config.serverUrl)
    private val audioRecorder = AudioRecorder(config.sampleRate, config.chunkDurationMs)
    
    private val _state = MutableStateFlow<DetectorState>(DetectorState.Uninitialized)
    val state: StateFlow<DetectorState> = _state

    private var detectionJob: Job? = null

    companion object {
        private const val TAG = "WakeWordDetector"
    }

    /**
     * Initialize the detector
     */
    suspend fun initialize(
        onProgress: ((TrainingStatus) -> Unit)? = null,
    ) {
        _state.value = DetectorState.Initializing
        
        try {
            // Check server health
            if (!client.isHealthy()) {
                throw IOException("Server is not reachable")
            }

            // Ensure model exists
            client.ensureModel(
                config.word,
                onProgress = onProgress,
            )

            _state.value = DetectorState.Ready
            Log.d(TAG, "Detector initialized for '${config.word}'")
        } catch (e: Exception) {
            Log.e(TAG, "Initialization failed: ${e.message}")
            _state.value = DetectorState.Error(e.message ?: "Unknown error")
            throw e
        }
    }

    /**
     * Start listening for wake word
     */
    fun startListening(
        onWakeWord: (Float) -> Unit,
        onError: ((Exception) -> Unit)? = null,
    ): Boolean {
        if (_state.value != DetectorState.Ready) {
            Log.w(TAG, "Detector not ready")
            return false
        }

        val started = audioRecorder.start(context) { audioBytes ->
            detectionJob = CoroutineScope(Dispatchers.IO).launch {
                try {
                    val result = client.predict(config.word, audioBytes)
                    
                    if (result.detected && result.confidence >= config.threshold) {
                        Log.d(TAG, "Wake word detected! Confidence: ${result.confidence}")
                        withContext(Dispatchers.Main) {
                            onWakeWord(result.confidence)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Prediction error: ${e.message}")
                    withContext(Dispatchers.Main) {
                        onError?.invoke(e)
                    }
                }
            }
        }

        if (started) {
            _state.value = DetectorState.Listening
            Log.d(TAG, "Started listening for '${config.word}'")
        }

        return started
    }

    /**
     * Stop listening
     */
    fun stopListening() {
        audioRecorder.stop()
        detectionJob?.cancel()
        
        if (_state.value == DetectorState.Listening) {
            _state.value = DetectorState.Ready
        }
        
        Log.d(TAG, "Stopped listening")
    }

    /**
     * Release resources
     */
    fun release() {
        stopListening()
        _state.value = DetectorState.Uninitialized
    }
}

// ============================================================================
// Local Detector (ONNX Runtime)
// ============================================================================

/**
 * Local wake word detector using ONNX Runtime
 * 
 * For local inference, add these dependencies:
 * implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
 */
class LocalWakeWordDetector(
    private val context: Context,
    private val config: WakeWordConfig,
) {
    private var modelBytes: ByteArray? = null
    // private var ortSession: OrtSession? = null  // Uncomment with ONNX Runtime
    
    private val audioRecorder = AudioRecorder(config.sampleRate, config.chunkDurationMs)
    private val client = WakeWordClient(config.serverUrl)
    
    private val _state = MutableStateFlow<DetectorState>(DetectorState.Uninitialized)
    val state: StateFlow<DetectorState> = _state

    companion object {
        private const val TAG = "LocalWakeWordDetector"
    }

    /**
     * Initialize with local model
     */
    suspend fun initialize() {
        _state.value = DetectorState.Initializing

        try {
            // Try to load cached model
            val modelFile = File(context.filesDir, "${config.word}_model.onnx")
            
            if (modelFile.exists()) {
                modelBytes = modelFile.readBytes()
            } else {
                // Download from server
                Log.d(TAG, "Downloading model...")
                modelBytes = client.downloadModel(config.word, "onnx")
                modelFile.writeBytes(modelBytes!!)
            }

            // Initialize ONNX Runtime
            // Uncomment when using ONNX Runtime:
            // val ortEnv = OrtEnvironment.getEnvironment()
            // ortSession = ortEnv.createSession(modelBytes)

            _state.value = DetectorState.Ready
            Log.d(TAG, "Local detector initialized")
        } catch (e: Exception) {
            _state.value = DetectorState.Error(e.message ?: "Unknown error")
            throw e
        }
    }

    /**
     * Start listening (local inference)
     */
    fun startListening(
        onWakeWord: (Float) -> Unit,
        onError: ((Exception) -> Unit)? = null,
    ): Boolean {
        if (_state.value != DetectorState.Ready) {
            return false
        }

        val started = audioRecorder.start(context) { audioBytes ->
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    val confidence = runLocalInference(audioBytes)
                    
                    if (confidence >= config.threshold) {
                        Log.d(TAG, "Wake word detected locally! Confidence: $confidence")
                        withContext(Dispatchers.Main) {
                            onWakeWord(confidence)
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Local inference error: ${e.message}")
                    withContext(Dispatchers.Main) {
                        onError?.invoke(e)
                    }
                }
            }
        }

        if (started) {
            _state.value = DetectorState.Listening
        }

        return started
    }

    private fun runLocalInference(audioBytes: ByteArray): Float {
        // TODO: Implement MFCC extraction and ONNX inference
        // 
        // val mfcc = extractMFCC(audioBytes)
        // val inputTensor = OnnxTensor.createTensor(ortEnv, mfcc)
        // val results = ortSession?.run(mapOf("input" to inputTensor))
        // return results?.get(0)?.value as Float
        
        return 0f  // Placeholder
    }

    fun stopListening() {
        audioRecorder.stop()
        if (_state.value == DetectorState.Listening) {
            _state.value = DetectorState.Ready
        }
    }

    fun release() {
        stopListening()
        // ortSession?.close()
        _state.value = DetectorState.Uninitialized
    }
}
