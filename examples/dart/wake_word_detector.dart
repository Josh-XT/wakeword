/// WakeWord Detector for Flutter/Dart
///
/// A cross-platform wake word detection library that works with
/// the WakeWord server for training and can run locally using ONNX.
///
/// Example usage:
/// ```dart
/// final detector = WakeWordDetector(
///   serverUrl: 'http://localhost:8000',
///   word: 'jarvis',
/// );
///
/// await detector.initialize();
/// detector.startListening(onWakeWord: () => print('Wake word detected!'));
/// ```

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

// For local inference, you'll need onnxruntime_flutter package
// import 'package:onnxruntime_flutter/onnxruntime_flutter.dart';

/// Configuration for wake word detection
class WakeWordConfig {
  final String word;
  final String serverUrl;
  final double threshold;
  final int sampleRate;
  final double chunkDuration;

  const WakeWordConfig({
    required this.word,
    this.serverUrl = 'http://localhost:8000',
    this.threshold = 0.5,
    this.sampleRate = 16000,
    this.chunkDuration = 1.5,
  });
}

/// Status of a training job
class TrainingJobStatus {
  final String jobId;
  final String word;
  final String status;
  final double progress;
  final String currentStage;
  final String? errorMessage;
  final String? estimatedCompletion;
  final Map<String, dynamic> metrics;

  TrainingJobStatus({
    required this.jobId,
    required this.word,
    required this.status,
    required this.progress,
    required this.currentStage,
    this.errorMessage,
    this.estimatedCompletion,
    this.metrics = const {},
  });

  factory TrainingJobStatus.fromJson(Map<String, dynamic> json) {
    return TrainingJobStatus(
      jobId: json['job_id'] ?? '',
      word: json['word'] ?? '',
      status: json['status'] ?? '',
      progress: (json['progress'] ?? 0).toDouble(),
      currentStage: json['current_stage'] ?? '',
      errorMessage: json['error_message'],
      estimatedCompletion: json['estimated_completion'],
      metrics: json['metrics'] ?? {},
    );
  }

  bool get isComplete => status == 'completed';
  bool get isFailed => status == 'failed';
  bool get isInProgress => !isComplete && !isFailed && status != 'cancelled';
}

/// Information about an available model
class ModelInfo {
  final String word;
  final String directory;
  final Map<String, dynamic> config;
  final Map<String, dynamic> files;
  final String createdAt;

  ModelInfo({
    required this.word,
    required this.directory,
    required this.config,
    required this.files,
    required this.createdAt,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      word: json['word'] ?? '',
      directory: json['directory'] ?? '',
      config: json['config'] ?? {},
      files: json['files'] ?? {},
      createdAt: json['created_at'] ?? '',
    );
  }
}

/// Result of wake word prediction
class PredictionResult {
  final bool detected;
  final double confidence;
  final String word;

  PredictionResult({
    required this.detected,
    required this.confidence,
    required this.word,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      detected: json['detected'] ?? false,
      confidence: (json['confidence'] ?? 0).toDouble(),
      word: json['word'] ?? '',
    );
  }
}

/// Client for interacting with the WakeWord server
class WakeWordClient {
  final String serverUrl;
  final http.Client _client;

  WakeWordClient({
    this.serverUrl = 'http://localhost:8000',
    http.Client? client,
  }) : _client = client ?? http.Client();

  /// Check if the server is healthy
  Future<bool> isHealthy() async {
    try {
      final response = await _client.get(Uri.parse('$serverUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// List all available models
  Future<List<ModelInfo>> listModels() async {
    final response = await _client.get(Uri.parse('$serverUrl/models'));

    if (response.statusCode != 200) {
      throw Exception('Failed to list models: ${response.body}');
    }

    final data = jsonDecode(response.body);
    final models =
        (data['models'] as List).map((m) => ModelInfo.fromJson(m)).toList();

    return models;
  }

  /// Check if a model exists for a word
  Future<bool> modelExists(String word) async {
    try {
      final response = await _client.get(
        Uri.parse('$serverUrl/models/$word/config'),
      );
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Request training for a new wake word model
  Future<TrainingJobStatus> requestTraining({
    required String word,
    int sampleCount = 500,
    int epochs = 50,
    int batchSize = 32,
  }) async {
    final response = await _client.post(
      Uri.parse('$serverUrl/train'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'word': word,
        'sample_count': sampleCount,
        'epochs': epochs,
        'batch_size': batchSize,
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to request training: ${response.body}');
    }

    return TrainingJobStatus.fromJson(jsonDecode(response.body));
  }

  /// Get the status of a training job
  Future<TrainingJobStatus> getJobStatus(String jobId) async {
    final response = await _client.get(Uri.parse('$serverUrl/jobs/$jobId'));

    if (response.statusCode != 200) {
      throw Exception('Failed to get job status: ${response.body}');
    }

    return TrainingJobStatus.fromJson(jsonDecode(response.body));
  }

  /// Wait for training to complete
  Future<TrainingJobStatus> waitForTraining(
    String jobId, {
    Duration pollInterval = const Duration(seconds: 10),
    Duration timeout = const Duration(minutes: 30),
    void Function(TrainingJobStatus)? onProgress,
  }) async {
    final deadline = DateTime.now().add(timeout);

    while (DateTime.now().isBefore(deadline)) {
      final status = await getJobStatus(jobId);

      if (onProgress != null) {
        onProgress(status);
      }

      if (status.isComplete) {
        return status;
      }

      if (status.isFailed) {
        throw Exception('Training failed: ${status.errorMessage}');
      }

      await Future.delayed(pollInterval);
    }

    throw TimeoutException('Training timed out');
  }

  /// Download a model file
  Future<Uint8List> downloadModel(String word, {String format = 'onnx'}) async {
    final response = await _client.get(
      Uri.parse('$serverUrl/models/$word?format=$format'),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to download model: ${response.body}');
    }

    return response.bodyBytes;
  }

  /// Download model to a file
  Future<File> downloadModelToFile(
    String word, {
    String format = 'onnx',
    String? directory,
  }) async {
    final bytes = await downloadModel(word, format: format);

    final dir = directory ?? (await getApplicationDocumentsDirectory()).path;
    final file = File('$dir/${word}_model.$format');
    await file.writeAsBytes(bytes);

    return file;
  }

  /// Ensure a model exists, training if necessary
  Future<void> ensureModel(
    String word, {
    int sampleCount = 500,
    int epochs = 50,
    void Function(TrainingJobStatus)? onProgress,
  }) async {
    if (await modelExists(word)) {
      return;
    }

    final job = await requestTraining(
      word: word,
      sampleCount: sampleCount,
      epochs: epochs,
    );

    if (job.isComplete) {
      return;
    }

    await waitForTraining(job.jobId, onProgress: onProgress);
  }

  /// Predict if audio contains the wake word (server-side)
  Future<PredictionResult> predict(String word, Uint8List audioBytes) async {
    final audioBase64 = base64Encode(audioBytes);

    final response = await _client.post(
      Uri.parse('$serverUrl/predict/$word'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'audio_base64': audioBase64}),
    );

    if (response.statusCode != 200) {
      throw Exception('Prediction failed: ${response.body}');
    }

    return PredictionResult.fromJson(jsonDecode(response.body));
  }

  void dispose() {
    _client.close();
  }
}

/// Wake word detector with microphone input
///
/// This is a basic structure - actual implementation requires
/// platform-specific audio recording (e.g., flutter_sound, record)
abstract class WakeWordDetector {
  final WakeWordConfig config;
  final WakeWordClient client;

  bool _isListening = false;

  WakeWordDetector({required this.config, WakeWordClient? client})
    : client = client ?? WakeWordClient(serverUrl: config.serverUrl);

  bool get isListening => _isListening;

  /// Initialize the detector (download model, set up audio, etc.)
  Future<void> initialize();

  /// Start listening for the wake word
  Future<void> startListening({
    required void Function() onWakeWord,
    void Function(double confidence)? onPrediction,
    void Function(Object error)? onError,
  });

  /// Stop listening
  Future<void> stopListening();

  /// Dispose resources
  void dispose() {
    stopListening();
    client.dispose();
  }
}

/// Server-based wake word detector
///
/// Sends audio to server for prediction.
/// Simpler to implement but requires network connectivity.
class ServerWakeWordDetector extends WakeWordDetector {
  StreamSubscription? _audioSubscription;

  ServerWakeWordDetector({
    required WakeWordConfig config,
    WakeWordClient? client,
  }) : super(config: config, client: client);

  @override
  Future<void> initialize() async {
    // Ensure model exists on server
    await client.ensureModel(
      config.word,
      onProgress: (status) {
        print(
          'Training progress: ${status.progress}% - ${status.currentStage}',
        );
      },
    );
  }

  @override
  Future<void> startListening({
    required void Function() onWakeWord,
    void Function(double confidence)? onPrediction,
    void Function(Object error)? onError,
  }) async {
    if (_isListening) return;
    _isListening = true;

    // TODO: Implement actual audio recording using platform-specific packages
    // Example with flutter_sound or record package:
    //
    // final recorder = FlutterSoundRecorder();
    // await recorder.openRecorder();
    //
    // Timer.periodic(Duration(milliseconds: (config.chunkDuration * 1000).toInt()), (timer) async {
    //   if (!_isListening) {
    //     timer.cancel();
    //     return;
    //   }
    //
    //   try {
    //     final audioBytes = await recorder.getAudioChunk();
    //     final result = await client.predict(config.word, audioBytes);
    //
    //     if (onPrediction != null) {
    //       onPrediction(result.confidence);
    //     }
    //
    //     if (result.detected && result.confidence >= config.threshold) {
    //       onWakeWord();
    //     }
    //   } catch (e) {
    //     if (onError != null) {
    //       onError(e);
    //     }
    //   }
    // });

    print('Listening for wake word "${config.word}"...');
    print(
      'Note: Audio recording not implemented - use flutter_sound or record package',
    );
  }

  @override
  Future<void> stopListening() async {
    _isListening = false;
    await _audioSubscription?.cancel();
    _audioSubscription = null;
  }
}

/// Local wake word detector using ONNX Runtime
///
/// Runs inference locally without network connectivity.
/// Requires onnxruntime_flutter package.
class LocalWakeWordDetector extends WakeWordDetector {
  File? _modelFile;
  // OrtSession? _session;  // Uncomment when using onnxruntime_flutter

  LocalWakeWordDetector({
    required WakeWordConfig config,
    WakeWordClient? client,
  }) : super(config: config, client: client);

  @override
  Future<void> initialize() async {
    // Download model if needed
    _modelFile = await client.downloadModelToFile(config.word, format: 'onnx');

    // Initialize ONNX Runtime
    // Uncomment when using onnxruntime_flutter:
    // await OrtEnv.instance.init();
    // _session = OrtSession.fromFile(_modelFile!.path);

    print('Local model loaded from: ${_modelFile!.path}');
  }

  @override
  Future<void> startListening({
    required void Function() onWakeWord,
    void Function(double confidence)? onPrediction,
    void Function(Object error)? onError,
  }) async {
    if (_isListening) return;
    _isListening = true;

    // TODO: Implement local inference with ONNX Runtime
    // Example:
    //
    // Timer.periodic(Duration(milliseconds: (config.chunkDuration * 1000).toInt()), (timer) async {
    //   if (!_isListening) {
    //     timer.cancel();
    //     return;
    //   }
    //
    //   try {
    //     final audioData = await getAudioChunk();
    //     final mfcc = extractMFCC(audioData);
    //
    //     final inputs = {'input': OrtValueTensor.fromList(mfcc, [1, 40, 150])};
    //     final outputs = await _session!.run(inputs);
    //     final confidence = outputs['output']!.value[0];
    //
    //     if (onPrediction != null) {
    //       onPrediction(confidence);
    //     }
    //
    //     if (confidence >= config.threshold) {
    //       onWakeWord();
    //     }
    //   } catch (e) {
    //     if (onError != null) {
    //       onError(e);
    //     }
    //   }
    // });

    print('Listening for wake word "${config.word}" (local mode)...');
    print(
      'Note: ONNX inference not implemented - use onnxruntime_flutter package',
    );
  }

  @override
  Future<void> stopListening() async {
    _isListening = false;
  }

  @override
  void dispose() {
    // _session?.release();
    super.dispose();
  }
}

// ============================================================================
// Example Usage
// ============================================================================

void main() async {
  // Example: Using server-based detection
  final config = WakeWordConfig(
    word: 'jarvis',
    serverUrl: 'http://localhost:8000',
    threshold: 0.5,
  );

  final detector = ServerWakeWordDetector(config: config);

  try {
    print('Initializing wake word detector...');
    await detector.initialize();

    print('Starting to listen...');
    await detector.startListening(
      onWakeWord: () {
        print('🎤 Wake word detected!');
      },
      onPrediction: (confidence) {
        if (confidence > 0.3) {
          print('Confidence: ${(confidence * 100).toStringAsFixed(1)}%');
        }
      },
      onError: (error) {
        print('Error: $error');
      },
    );

    // Keep running for demo
    await Future.delayed(Duration(minutes: 5));
  } finally {
    detector.dispose();
  }
}
