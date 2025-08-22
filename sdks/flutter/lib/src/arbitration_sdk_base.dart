/// Main SDK class for Flutter Arbitration Platform
/// 
/// This is the primary interface for interacting with the Arbitration SDK.
/// It provides methods for text analysis, document processing, and configuration management.

import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'models/analysis_result.dart';
import 'models/analysis_request.dart';
import 'models/batch_analysis.dart';
import 'models/configuration.dart';
import 'models/user_preferences.dart';
import 'models/analytics_event.dart';
import 'services/network_service.dart';
import 'services/storage_service.dart';
import 'services/ocr_service.dart';
import 'services/biometric_service.dart';
import 'services/offline_sync_service.dart';
import 'services/analytics_service.dart';
import 'platform/arbitration_platform_interface.dart';
import 'exceptions/arbitration_exception.dart';
import 'utils/validation.dart';

/// Main ArbitrationSDK class
class ArbitrationSDK {
  static ArbitrationSDK? _instance;
  static final Completer<ArbitrationSDK> _initCompleter = Completer<ArbitrationSDK>();
  
  late final NetworkService _networkService;
  late final StorageService _storageService;
  late final OCRService _ocrService;
  late final BiometricService _biometricService;
  late final OfflineSyncService _offlineSyncService;
  late final AnalyticsService _analyticsService;
  
  SDKConfiguration _configuration = const SDKConfiguration();
  bool _isInitialized = false;
  
  // Event streams
  final StreamController<AnalysisProgress> _analysisProgressController = 
      StreamController<AnalysisProgress>.broadcast();
  final StreamController<AnalysisResult> _analysisResultController = 
      StreamController<AnalysisResult>.broadcast();
  final StreamController<ArbitrationException> _errorController = 
      StreamController<ArbitrationException>.broadcast();
  final StreamController<bool> _networkStatusController = 
      StreamController<bool>.broadcast();

  ArbitrationSDK._internal();

  /// Get the singleton instance of ArbitrationSDK
  static ArbitrationSDK get instance {
    _instance ??= ArbitrationSDK._internal();
    return _instance!;
  }

  /// Initialize the SDK with optional configuration
  static Future<ArbitrationSDK> initialize([SDKConfiguration? config]) async {
    if (_initCompleter.isCompleted) {
      return _initCompleter.future;
    }

    final sdk = ArbitrationSDK.instance;
    await sdk._initialize(config);
    
    if (!_initCompleter.isCompleted) {
      _initCompleter.complete(sdk);
    }
    
    return sdk;
  }

  /// Private initialization method
  Future<void> _initialize(SDKConfiguration? config) async {
    if (_isInitialized) return;

    // Update configuration
    if (config != null) {
      _configuration = config;
    }

    // Validate configuration
    final validationResult = ValidationUtils.validateConfiguration(_configuration);
    if (!validationResult.isValid) {
      throw ArbitrationException.invalidConfiguration(
        'Invalid configuration: ${validationResult.errors.join(', ')}'
      );
    }

    try {
      // Initialize services
      _networkService = NetworkService(_configuration);
      _storageService = StorageService();
      _ocrService = OCRService();
      _biometricService = BiometricService();
      _offlineSyncService = OfflineSyncService(_storageService, _networkService);
      _analyticsService = AnalyticsService(_configuration, _storageService);

      // Initialize storage
      await _storageService.initialize();

      // Initialize platform interface
      await ArbitrationPlatformInterface.instance.configure(_configuration.toMap());

      // Setup event listeners
      _setupEventListeners();

      _isInitialized = true;
      debugPrint('ArbitrationSDK initialized successfully');
    } catch (e) {
      throw ArbitrationException.initializationFailed(
        'Failed to initialize SDK: $e'
      );
    }
  }

  /// Update SDK configuration
  void updateConfiguration(SDKConfiguration config) {
    _configuration = config;
    _networkService.updateConfiguration(config);
    _analyticsService.updateConfiguration(config);
    
    // Update platform interface
    ArbitrationPlatformInterface.instance.configure(config.toMap());
  }

  /// Get current configuration
  SDKConfiguration get configuration => _configuration;

  /// Check if SDK is initialized
  bool get isInitialized => _isInitialized;

  // Analysis Methods

  /// Analyze text for arbitration clauses
  Future<AnalysisResult> analyzeText(String text) async {
    _ensureInitialized();
    
    final request = AnalysisRequest(text: text);
    final validationResult = ValidationUtils.validateAnalysisRequest(request);
    
    if (!validationResult.isValid) {
      throw ArbitrationException.invalidInput(
        'Invalid analysis request: ${validationResult.errors.join(', ')}'
      );
    }

    try {
      _analysisProgressController.add(AnalysisProgress(0.0, 'Starting analysis...'));
      
      // Track analytics
      _analyticsService.trackEvent(AnalyticsEvent.analysisStarted());

      // Perform analysis
      final result = await _performAnalysis(text);

      // Save result locally
      await _storageService.saveAnalysisResult(result);

      // Track completion
      _analyticsService.trackEvent(AnalyticsEvent.analysisCompleted(result));

      _analysisResultController.add(result);
      _analysisProgressController.add(AnalysisProgress(1.0, 'Analysis complete'));

      return result;
    } catch (e) {
      final exception = e is ArbitrationException 
          ? e 
          : ArbitrationException.analysisFailure('Analysis failed: $e');
      
      _analyticsService.trackEvent(AnalyticsEvent.analysisFailed(exception.toString()));
      _errorController.add(exception);
      
      throw exception;
    }
  }

  /// Analyze document from file path
  Future<AnalysisResult> analyzeDocument(String filePath) async {
    _ensureInitialized();

    try {
      _analysisProgressController.add(AnalysisProgress(0.0, 'Processing document...'));

      // Extract text from document
      final extractedText = await _extractTextFromDocument(filePath);
      
      _analysisProgressController.add(AnalysisProgress(0.3, 'Text extracted, analyzing...'));

      // Analyze extracted text
      final result = await analyzeText(extractedText);

      _analyticsService.trackEvent(AnalyticsEvent.documentAnalyzed(filePath));

      return result;
    } catch (e) {
      final exception = ArbitrationException.documentProcessingFailed(
        'Failed to analyze document: $e'
      );
      _errorController.add(exception);
      throw exception;
    }
  }

  /// Analyze image with OCR
  Future<AnalysisResult> analyzeImage(String imagePath) async {
    _ensureInitialized();

    try {
      _analysisProgressController.add(AnalysisProgress(0.0, 'Processing image...'));

      // Extract text using OCR
      final ocrResult = await _ocrService.extractTextFromImage(imagePath);
      
      _analysisProgressController.add(AnalysisProgress(0.4, 'Text extracted, analyzing...'));

      // Analyze extracted text
      final result = await analyzeText(ocrResult.text);

      _analyticsService.trackEvent(AnalyticsEvent.imageAnalyzed(imagePath, ocrResult.confidence));

      return result;
    } catch (e) {
      final exception = ArbitrationException.ocrFailed('OCR analysis failed: $e');
      _errorController.add(exception);
      throw exception;
    }
  }

  /// Batch analyze multiple requests
  Future<BatchAnalysisResult> batchAnalyze(List<AnalysisRequest> requests) async {
    _ensureInitialized();

    if (requests.isEmpty) {
      throw ArbitrationException.invalidInput('No requests provided for batch analysis');
    }

    try {
      _analysisProgressController.add(AnalysisProgress(0.0, 'Starting batch analysis...'));

      final results = <AnalysisResult>[];
      final startTime = DateTime.now();

      for (int i = 0; i < requests.length; i++) {
        final progress = (i + 1) / requests.length;
        _analysisProgressController.add(
          AnalysisProgress(progress, 'Analyzing ${i + 1}/${requests.length}...')
        );

        final result = await _performAnalysis(requests[i].text);
        results.add(result);

        // Save individual results
        await _storageService.saveAnalysisResult(result);
      }

      final endTime = DateTime.now();
      final processingTime = endTime.difference(startTime);

      final summary = BatchSummary(
        totalDocuments: results.length,
        documentsWithArbitration: results.where((r) => r.hasArbitration).length,
        averageConfidence: results.map((r) => r.confidence).reduce((a, b) => a + b) / results.length,
        highRiskDocuments: results.where((r) => r.riskLevel == RiskLevel.high).length,
        processingErrors: 0,
      );

      final batchResult = BatchAnalysisResult(
        batchId: DateTime.now().millisecondsSinceEpoch.toString(),
        results: results,
        summary: summary,
        processingTime: processingTime,
      );

      _analyticsService.trackEvent(AnalyticsEvent.batchAnalysisCompleted(batchResult));

      return batchResult;
    } catch (e) {
      final exception = ArbitrationException.batchAnalysisFailed('Batch analysis failed: $e');
      _errorController.add(exception);
      throw exception;
    }
  }

  // OCR Methods

  /// Extract text from image using OCR
  Future<OCRResult> extractTextFromImage(String imagePath) async {
    _ensureInitialized();
    return await _ocrService.extractTextFromImage(imagePath);
  }

  /// Extract text from camera
  Future<OCRResult> extractTextFromCamera() async {
    _ensureInitialized();
    return await _ocrService.extractTextFromCamera();
  }

  // Biometric Authentication

  /// Check if biometric authentication is available
  Future<bool> isBiometricAvailable() async {
    return await _biometricService.isAvailable();
  }

  /// Get available biometric types
  Future<List<BiometricType>> getAvailableBiometrics() async {
    return await _biometricService.getAvailableBiometrics();
  }

  /// Authenticate with biometrics
  Future<bool> authenticateWithBiometrics({
    required String localizedFallbackTitle,
    String? reason,
  }) async {
    return await _biometricService.authenticate(
      localizedFallbackTitle: localizedFallbackTitle,
      reason: reason ?? 'Please authenticate to access the application',
    );
  }

  // Offline & Sync

  /// Enable or disable offline mode
  void enableOfflineMode(bool enabled) {
    _configuration = _configuration.copyWith(enableOfflineMode: enabled);
    _offlineSyncService.setEnabled(enabled);
  }

  /// Sync offline data
  Future<void> syncOfflineData() async {
    if (!_configuration.enableOfflineMode) {
      throw ArbitrationException.offlineModeDisabled('Offline mode is not enabled');
    }

    await _offlineSyncService.sync();
  }

  /// Get offline sync status
  Future<OfflineSyncStatus> getSyncStatus() async {
    return await _offlineSyncService.getStatus();
  }

  // User Preferences

  /// Save user preferences
  Future<void> saveUserPreferences(UserPreferences preferences) async {
    await _storageService.saveUserPreferences(preferences);
  }

  /// Get user preferences
  Future<UserPreferences?> getUserPreferences() async {
    return await _storageService.getUserPreferences();
  }

  // Analytics

  /// Track custom event
  void trackEvent(AnalyticsEvent event) {
    _analyticsService.trackEvent(event);
  }

  /// Get analytics statistics
  Future<Map<String, dynamic>> getAnalyticsStatistics() async {
    return await _analyticsService.getStatistics();
  }

  // Storage & Cache

  /// Clear all cached data
  Future<void> clearCache() async {
    await _storageService.clearCache();
  }

  /// Get storage statistics
  Future<StorageStatistics> getStorageStatistics() async {
    return await _storageService.getStatistics();
  }

  /// Export analysis data
  Future<String> exportAnalysisData([String format = 'json']) async {
    return await _storageService.exportData(format);
  }

  // Event Streams

  /// Stream of analysis progress updates
  Stream<AnalysisProgress> get analysisProgressStream => _analysisProgressController.stream;

  /// Stream of analysis results
  Stream<AnalysisResult> get analysisResultStream => _analysisResultController.stream;

  /// Stream of errors
  Stream<ArbitrationException> get errorStream => _errorController.stream;

  /// Stream of network status changes
  Stream<bool> get networkStatusStream => _networkStatusController.stream;

  // Private Methods

  void _ensureInitialized() {
    if (!_isInitialized) {
      throw ArbitrationException.notInitialized('SDK not initialized. Call initialize() first.');
    }
  }

  Future<AnalysisResult> _performAnalysis(String text) async {
    // Implementation would call platform-specific analysis
    // This is a simplified version
    
    _analysisProgressController.add(AnalysisProgress(0.2, 'Preprocessing text...'));
    
    // Platform channel call
    final result = await ArbitrationPlatformInterface.instance.analyzeText(text);
    
    _analysisProgressController.add(AnalysisProgress(0.8, 'Finalizing results...'));
    
    return AnalysisResult.fromMap(result);
  }

  Future<String> _extractTextFromDocument(String filePath) async {
    // Implementation would extract text from various document formats
    // This is a placeholder
    final file = File(filePath);
    if (!file.existsSync()) {
      throw ArbitrationException.fileNotFound('File not found: $filePath');
    }

    // For now, assume it's a text file
    return await file.readAsString();
  }

  void _setupEventListeners() {
    // Setup platform event listeners
    ArbitrationPlatformInterface.instance.onAnalysisProgress.listen((progress) {
      _analysisProgressController.add(AnalysisProgress(progress, ''));
    });

    ArbitrationPlatformInterface.instance.onNetworkStatusChanged.listen((isConnected) {
      _networkStatusController.add(isConnected);
    });
  }

  /// Dispose of resources
  void dispose() {
    _analysisProgressController.close();
    _analysisResultController.close();
    _errorController.close();
    _networkStatusController.close();
    
    _storageService.dispose();
    _offlineSyncService.dispose();
    _analyticsService.dispose();
  }
}

/// Analysis progress information
class AnalysisProgress {
  final double progress;
  final String message;

  const AnalysisProgress(this.progress, this.message);

  @override
  String toString() => 'AnalysisProgress(progress: $progress, message: $message)';
}