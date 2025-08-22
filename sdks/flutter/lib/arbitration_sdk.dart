/// Flutter SDK for Arbitration Platform
/// 
/// Provides comprehensive arbitration clause detection and analysis
/// with support for text analysis, document scanning, OCR, and AR visualization.
library arbitration_sdk;

// Core SDK
export 'src/arbitration_sdk_base.dart';
export 'src/arbitration_detector.dart';

// Models
export 'src/models/analysis_result.dart';
export 'src/models/analysis_request.dart';
export 'src/models/batch_analysis.dart';
export 'src/models/configuration.dart';
export 'src/models/user_preferences.dart';
export 'src/models/analytics_event.dart';

// Services
export 'src/services/network_service.dart';
export 'src/services/storage_service.dart';
export 'src/services/ocr_service.dart';
export 'src/services/biometric_service.dart';
export 'src/services/offline_sync_service.dart';
export 'src/services/analytics_service.dart';

// State Management (Riverpod)
export 'src/providers/arbitration_provider.dart';
export 'src/providers/offline_provider.dart';
export 'src/providers/preferences_provider.dart';
export 'src/providers/analytics_provider.dart';

// UI Components
export 'src/widgets/document_analyzer_widget.dart';
export 'src/widgets/analysis_result_widget.dart';
export 'src/widgets/camera_scanner_widget.dart';
export 'src/widgets/confidence_chart.dart';
export 'src/widgets/risk_level_indicator.dart';
export 'src/widgets/keyword_highlight_widget.dart';

// Utilities
export 'src/utils/validation.dart';
export 'src/utils/formatting.dart';
export 'src/utils/constants.dart';
export 'src/utils/extensions.dart';

// Exceptions
export 'src/exceptions/arbitration_exception.dart';

// Platform Channels
export 'src/platform/arbitration_platform_interface.dart';
export 'src/platform/method_channel_arbitration.dart';

// Constants
const String sdkVersion = '1.0.0';
const String sdkName = 'ArbitrationSDK Flutter';

/// Supported document types for analysis
const List<String> supportedDocumentTypes = [
  'application/pdf',
  'text/plain', 
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'image/jpeg',
  'image/png', 
  'image/tiff'
];

/// Default SDK configuration
const Map<String, dynamic> defaultConfiguration = {
  'apiBaseUrl': 'https://api.arbitration-platform.com',
  'confidenceThreshold': 0.5,
  'enableOfflineMode': true,
  'enableAnalytics': true,
  'cacheSize': 100,
  'networkTimeout': 30000,
};