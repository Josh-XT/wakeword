/*
 * ESP32 Wake Word Detector
 * 
 * Uses TensorFlow Lite Micro to run wake word detection on ESP32.
 * 
 * Hardware requirements:
 * - ESP32 (ESP32-WROOM-32 or better)
 * - I2S microphone (e.g., INMP441, SPH0645)
 * - Optional: LED for wake indication
 * 
 * Setup:
 * 1. Download your trained .tflite model from the WakeWord server
 * 2. Convert to C array using xxd: xxd -i model.tflite > model_data.h
 * 3. Update MODEL_DATA and MODEL_DATA_LEN in this file
 * 4. Configure I2S pins for your microphone
 * 5. Flash to ESP32
 */

#include <Arduino.h>
#include <driver/i2s.h>

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Include your model data (generated from .tflite file)
// #include "model_data.h"

// ============================================================================
// Configuration
// ============================================================================

// Wake word configuration
#define WAKE_WORD "jarvis"
#define DETECTION_THRESHOLD 0.5f

// Audio configuration
#define SAMPLE_RATE 16000
#define AUDIO_LENGTH_MS 1500
#define AUDIO_LENGTH_SAMPLES (SAMPLE_RATE * AUDIO_LENGTH_MS / 1000)

// I2S configuration (adjust for your microphone)
#define I2S_PORT I2S_NUM_0
#define I2S_BCLK_PIN 26
#define I2S_WS_PIN 25
#define I2S_DATA_PIN 22

// MFCC configuration (must match training)
#define N_MFCC 40
#define N_FFT 512
#define HOP_LENGTH 160
#define WIN_LENGTH 400
#define MAX_FRAMES 150

// TensorFlow Lite arena size
#define TENSOR_ARENA_SIZE (50 * 1024)  // 50KB

// LED pin for wake indication
#define WAKE_LED_PIN 2

// ============================================================================
// Globals
// ============================================================================

// Audio buffer
int16_t audio_buffer[AUDIO_LENGTH_SAMPLES];
int audio_buffer_index = 0;

// MFCC feature buffer
float mfcc_features[N_MFCC * MAX_FRAMES];

// TensorFlow Lite
uint8_t tensor_arena[TENSOR_ARENA_SIZE];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Placeholder model data - replace with your actual model
// Generate this by running: xxd -i model.tflite > model_data.h
const unsigned char model_data[] = {
    // Your model bytes here
    0x00  // Placeholder
};
const unsigned int model_data_len = 1;

// ============================================================================
// I2S Microphone Setup
// ============================================================================

void setup_i2s() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 1024,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCLK_PIN,
        .ws_io_num = I2S_WS_PIN,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_DATA_PIN
    };
    
    ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pin_config));
}

// ============================================================================
// MFCC Feature Extraction (Simplified)
// ============================================================================

// Pre-computed Mel filterbank (simplified - in production use proper filterbank)
// This is a placeholder - real implementation needs proper DSP

void compute_mfcc(int16_t* audio, int audio_len, float* mfcc_out) {
    // NOTE: This is a simplified placeholder. For production:
    // 1. Use a proper FFT library (e.g., esp-dsp)
    // 2. Implement proper Mel filterbank
    // 3. Apply DCT for MFCC computation
    
    // For now, we'll use a very simplified approach
    // In production, consider using:
    // - esp-dsp library for FFT
    // - Pre-computed Mel filterbank weights
    // - Fixed-point arithmetic for efficiency
    
    int num_frames = (audio_len - WIN_LENGTH) / HOP_LENGTH + 1;
    if (num_frames > MAX_FRAMES) num_frames = MAX_FRAMES;
    
    // Zero output buffer
    memset(mfcc_out, 0, N_MFCC * MAX_FRAMES * sizeof(float));
    
    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * HOP_LENGTH;
        
        // Simple energy-based features as placeholder
        // Real implementation needs FFT + Mel filterbank + DCT
        float frame_energy = 0;
        for (int i = 0; i < WIN_LENGTH && (start + i) < audio_len; i++) {
            float sample = audio[start + i] / 32768.0f;
            frame_energy += sample * sample;
        }
        frame_energy = logf(frame_energy + 1e-10f);
        
        // Fill MFCC coefficients (placeholder)
        for (int m = 0; m < N_MFCC; m++) {
            mfcc_out[m * MAX_FRAMES + frame] = frame_energy * (1.0f / (m + 1));
        }
    }
}

// ============================================================================
// TensorFlow Lite Setup
// ============================================================================

bool setup_tflite() {
    // Load model
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        return false;
    }
    
    // Create interpreter
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return false;
    }
    
    // Get input/output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    Serial.println("TFLite model loaded successfully");
    Serial.printf("Input shape: [%d, %d]\n", input->dims->data[1], input->dims->data[2]);
    Serial.printf("Arena used: %d bytes\n", interpreter->arena_used_bytes());
    
    return true;
}

// ============================================================================
// Wake Word Detection
// ============================================================================

float detect_wake_word() {
    // Extract MFCC features
    compute_mfcc(audio_buffer, AUDIO_LENGTH_SAMPLES, mfcc_features);
    
    // Copy features to input tensor
    for (int i = 0; i < N_MFCC * MAX_FRAMES; i++) {
        input->data.f[i] = mfcc_features[i];
    }
    
    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Inference failed!");
        return 0.0f;
    }
    
    // Get output
    float confidence = output->data.f[0];
    return confidence;
}

// ============================================================================
// Audio Capture
// ============================================================================

void capture_audio() {
    size_t bytes_read;
    int16_t samples[512];
    
    // Read samples from I2S
    i2s_read(I2S_PORT, samples, sizeof(samples), &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int16_t);
    
    // Add to circular buffer
    for (int i = 0; i < samples_read; i++) {
        audio_buffer[audio_buffer_index] = samples[i];
        audio_buffer_index = (audio_buffer_index + 1) % AUDIO_LENGTH_SAMPLES;
    }
}

void get_audio_snapshot(int16_t* dest) {
    // Copy audio buffer maintaining time order
    int start_idx = audio_buffer_index;
    for (int i = 0; i < AUDIO_LENGTH_SAMPLES; i++) {
        dest[i] = audio_buffer[(start_idx + i) % AUDIO_LENGTH_SAMPLES];
    }
}

// ============================================================================
// Main
// ============================================================================

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("\n=== ESP32 Wake Word Detector ===");
    Serial.printf("Wake word: %s\n", WAKE_WORD);
    Serial.printf("Threshold: %.2f\n", DETECTION_THRESHOLD);
    
    // Setup LED
    pinMode(WAKE_LED_PIN, OUTPUT);
    digitalWrite(WAKE_LED_PIN, LOW);
    
    // Setup I2S microphone
    Serial.println("Initializing I2S microphone...");
    setup_i2s();
    
    // Setup TensorFlow Lite
    Serial.println("Loading TFLite model...");
    if (!setup_tflite()) {
        Serial.println("Failed to initialize TFLite!");
        while (1) delay(1000);
    }
    
    Serial.println("Ready! Listening for wake word...\n");
}

void loop() {
    static unsigned long last_detection = 0;
    static int16_t snapshot[AUDIO_LENGTH_SAMPLES];
    
    // Capture audio continuously
    capture_audio();
    
    // Run detection every 500ms (with overlap)
    if (millis() - last_detection >= 500) {
        last_detection = millis();
        
        // Get audio snapshot
        get_audio_snapshot(snapshot);
        memcpy(audio_buffer, snapshot, sizeof(snapshot));
        
        // Run detection
        unsigned long start_time = micros();
        float confidence = detect_wake_word();
        unsigned long inference_time = micros() - start_time;
        
        // Check threshold
        if (confidence >= DETECTION_THRESHOLD) {
            Serial.printf("🎤 WAKE WORD DETECTED! (confidence: %.2f, time: %lu us)\n", 
                         confidence, inference_time);
            
            // Visual indication
            digitalWrite(WAKE_LED_PIN, HIGH);
            delay(500);
            digitalWrite(WAKE_LED_PIN, LOW);
            
            // Add cooldown to prevent repeated triggers
            delay(1000);
        } else if (confidence > 0.2) {
            // Debug: show near-misses
            Serial.printf("Near-miss: %.2f (inference: %lu us)\n", confidence, inference_time);
        }
    }
}

// ============================================================================
// Utility: Model Download Helper
// ============================================================================

/*
 * To download and convert your model:
 * 
 * 1. Download TFLite model from server:
 *    curl -o model.tflite "http://your-server:8000/models/jarvis?format=tflite"
 * 
 * 2. Convert to C array:
 *    xxd -i model.tflite > model_data.h
 * 
 * 3. Include in this file:
 *    #include "model_data.h"
 * 
 * 4. Update model_data and model_data_len references above
 */
