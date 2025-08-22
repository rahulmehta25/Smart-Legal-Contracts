/// Analysis result models for Flutter Arbitration SDK

import 'package:equatable/equatable.dart';
import 'package:json_annotation/json_annotation.dart';

part 'analysis_result.g.dart';

/// Main analysis result containing all arbitration detection information
@JsonSerializable()
class AnalysisResult extends Equatable {
  /// Unique identifier for this analysis
  final String id;
  
  /// Whether arbitration clause was detected
  final bool hasArbitration;
  
  /// Confidence score (0.0 to 1.0)
  final double confidence;
  
  /// List of detected keywords
  final List<KeywordMatch> keywordMatches;
  
  /// List of exclusion phrases found
  final List<KeywordMatch> exclusionMatches;
  
  /// Risk level assessment
  final RiskLevel riskLevel;
  
  /// Recommendations for the user
  final List<String> recommendations;
  
  /// Analysis metadata
  final AnalysisMetadata metadata;

  const AnalysisResult({
    required this.id,
    required this.hasArbitration,
    required this.confidence,
    required this.keywordMatches,
    required this.exclusionMatches,
    required this.riskLevel,
    required this.recommendations,
    required this.metadata,
  });

  /// Create from JSON map
  factory AnalysisResult.fromJson(Map<String, dynamic> json) => 
      _$AnalysisResultFromJson(json);
  
  /// Create from platform map
  factory AnalysisResult.fromMap(Map<String, dynamic> map) => 
      AnalysisResult.fromJson(map);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$AnalysisResultToJson(this);

  @override
  List<Object?> get props => [
    id,
    hasArbitration,
    confidence,
    keywordMatches,
    exclusionMatches,
    riskLevel,
    recommendations,
    metadata,
  ];

  /// Create copy with updated fields
  AnalysisResult copyWith({
    String? id,
    bool? hasArbitration,
    double? confidence,
    List<KeywordMatch>? keywordMatches,
    List<KeywordMatch>? exclusionMatches,
    RiskLevel? riskLevel,
    List<String>? recommendations,
    AnalysisMetadata? metadata,
  }) {
    return AnalysisResult(
      id: id ?? this.id,
      hasArbitration: hasArbitration ?? this.hasArbitration,
      confidence: confidence ?? this.confidence,
      keywordMatches: keywordMatches ?? this.keywordMatches,
      exclusionMatches: exclusionMatches ?? this.exclusionMatches,
      riskLevel: riskLevel ?? this.riskLevel,
      recommendations: recommendations ?? this.recommendations,
      metadata: metadata ?? this.metadata,
    );
  }

  @override
  String toString() {
    return 'AnalysisResult(id: $id, hasArbitration: $hasArbitration, confidence: $confidence, riskLevel: $riskLevel)';
  }
}

/// Keyword match information
@JsonSerializable()
class KeywordMatch extends Equatable {
  /// The matched keyword or phrase
  final String keyword;
  
  /// Text range where match was found (stored as "start:end")
  final String range;
  
  /// Context around the matched keyword
  final String context;
  
  /// Confidence score for this match (0.0 to 1.0)
  final double confidence;

  const KeywordMatch({
    required this.keyword,
    required this.range,
    required this.context,
    required this.confidence,
  });

  /// Create from JSON map
  factory KeywordMatch.fromJson(Map<String, dynamic> json) => 
      _$KeywordMatchFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$KeywordMatchToJson(this);

  @override
  List<Object?> get props => [keyword, range, context, confidence];

  /// Get start position of match
  int get startPosition {
    final parts = range.split(':');
    return parts.isNotEmpty ? int.tryParse(parts[0]) ?? 0 : 0;
  }

  /// Get end position of match
  int get endPosition {
    final parts = range.split(':');
    return parts.length > 1 ? int.tryParse(parts[1]) ?? 0 : 0;
  }

  @override
  String toString() {
    return 'KeywordMatch(keyword: $keyword, confidence: $confidence)';
  }
}

/// Risk level enumeration
enum RiskLevel {
  @JsonValue('minimal')
  minimal,
  
  @JsonValue('low')
  low,
  
  @JsonValue('medium')
  medium,
  
  @JsonValue('high')
  high;

  /// Display name for UI
  String get displayName {
    switch (this) {
      case RiskLevel.minimal:
        return 'Minimal Risk';
      case RiskLevel.low:
        return 'Low Risk';
      case RiskLevel.medium:
        return 'Medium Risk';
      case RiskLevel.high:
        return 'High Risk';
    }
  }

  /// Color hex code for UI
  String get colorHex {
    switch (this) {
      case RiskLevel.minimal:
        return '#4CAF50'; // Green
      case RiskLevel.low:
        return '#FFC107'; // Yellow
      case RiskLevel.medium:
        return '#FF9800'; // Orange
      case RiskLevel.high:
        return '#F44336'; // Red
    }
  }

  /// Get risk level from confidence score
  static RiskLevel fromConfidence(double confidence) {
    if (confidence >= 0.8) return RiskLevel.high;
    if (confidence >= 0.5) return RiskLevel.medium;
    if (confidence >= 0.2) return RiskLevel.low;
    return RiskLevel.minimal;
  }
}

/// Analysis metadata
@JsonSerializable()
class AnalysisMetadata extends Equatable {
  /// Length of analyzed text
  final int textLength;
  
  /// Processing time in milliseconds
  final int processingTimeMs;
  
  /// SDK version used for analysis
  final String sdkVersion;
  
  /// ML model version used
  final String modelVersion;
  
  /// Date and time of analysis
  @JsonKey(fromJson: _dateTimeFromJson, toJson: _dateTimeToJson)
  final DateTime analysisDate;

  const AnalysisMetadata({
    required this.textLength,
    required this.processingTimeMs,
    required this.sdkVersion,
    required this.modelVersion,
    required this.analysisDate,
  });

  /// Create from JSON map
  factory AnalysisMetadata.fromJson(Map<String, dynamic> json) => 
      _$AnalysisMetadataFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$AnalysisMetadataToJson(this);

  /// Processing time as Duration
  Duration get processingTime => Duration(milliseconds: processingTimeMs);

  @override
  List<Object?> get props => [
    textLength,
    processingTimeMs,
    sdkVersion,
    modelVersion,
    analysisDate,
  ];

  @override
  String toString() {
    return 'AnalysisMetadata(textLength: $textLength, processingTime: ${processingTime.inMilliseconds}ms)';
  }
}

/// OCR result
@JsonSerializable()
class OCRResult extends Equatable {
  /// Extracted text
  final String text;
  
  /// Overall confidence score
  final double confidence;
  
  /// Text blocks with positions
  final List<TextBlock> blocks;

  const OCRResult({
    required this.text,
    required this.confidence,
    required this.blocks,
  });

  /// Create from JSON map
  factory OCRResult.fromJson(Map<String, dynamic> json) => 
      _$OCRResultFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$OCRResultToJson(this);

  @override
  List<Object?> get props => [text, confidence, blocks];
}

/// Text block from OCR
@JsonSerializable()
class TextBlock extends Equatable {
  /// Text content
  final String text;
  
  /// Bounding box
  final BoundingBox boundingBox;
  
  /// Confidence score for this block
  final double confidence;

  const TextBlock({
    required this.text,
    required this.boundingBox,
    required this.confidence,
  });

  /// Create from JSON map
  factory TextBlock.fromJson(Map<String, dynamic> json) => 
      _$TextBlockFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$TextBlockToJson(this);

  @override
  List<Object?> get props => [text, boundingBox, confidence];
}

/// Bounding box for text blocks
@JsonSerializable()
class BoundingBox extends Equatable {
  final double x;
  final double y;
  final double width;
  final double height;

  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  /// Create from JSON map
  factory BoundingBox.fromJson(Map<String, dynamic> json) => 
      _$BoundingBoxFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$BoundingBoxToJson(this);

  @override
  List<Object?> get props => [x, y, width, height];
}

/// Storage statistics
@JsonSerializable()
class StorageStatistics extends Equatable {
  final int totalAnalyses;
  final int analysesWithArbitration;
  final int totalEvents;
  final int unsyncedEvents;
  final double cacheHitRate;
  final int databaseSizeBytes;

  const StorageStatistics({
    required this.totalAnalyses,
    required this.analysesWithArbitration,
    required this.totalEvents,
    required this.unsyncedEvents,
    required this.cacheHitRate,
    required this.databaseSizeBytes,
  });

  /// Create from JSON map
  factory StorageStatistics.fromJson(Map<String, dynamic> json) => 
      _$StorageStatisticsFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$StorageStatisticsToJson(this);

  @override
  List<Object?> get props => [
    totalAnalyses,
    analysesWithArbitration,
    totalEvents,
    unsyncedEvents,
    cacheHitRate,
    databaseSizeBytes,
  ];
}

/// Offline sync status
@JsonSerializable()
class OfflineSyncStatus extends Equatable {
  final bool isOnline;
  final int pendingUploads;
  @JsonKey(fromJson: _nullableDateTimeFromJson, toJson: _nullableDateTimeToJson)
  final DateTime? lastSyncTime;
  final bool syncInProgress;

  const OfflineSyncStatus({
    required this.isOnline,
    required this.pendingUploads,
    this.lastSyncTime,
    required this.syncInProgress,
  });

  /// Create from JSON map
  factory OfflineSyncStatus.fromJson(Map<String, dynamic> json) => 
      _$OfflineSyncStatusFromJson(json);

  /// Convert to JSON map
  Map<String, dynamic> toJson() => _$OfflineSyncStatusToJson(this);

  @override
  List<Object?> get props => [isOnline, pendingUploads, lastSyncTime, syncInProgress];
}

/// Biometric types
enum BiometricType {
  @JsonValue('none')
  none,
  
  @JsonValue('fingerprint')
  fingerprint,
  
  @JsonValue('face')
  face,
  
  @JsonValue('iris')
  iris;

  String get displayName {
    switch (this) {
      case BiometricType.none:
        return 'None';
      case BiometricType.fingerprint:
        return 'Fingerprint';
      case BiometricType.face:
        return 'Face Recognition';
      case BiometricType.iris:
        return 'Iris Recognition';
    }
  }
}

// Helper functions for DateTime serialization
DateTime _dateTimeFromJson(String dateTimeString) => DateTime.parse(dateTimeString);
String _dateTimeToJson(DateTime dateTime) => dateTime.toIso8601String();

DateTime? _nullableDateTimeFromJson(String? dateTimeString) => 
    dateTimeString != null ? DateTime.parse(dateTimeString) : null;
String? _nullableDateTimeToJson(DateTime? dateTime) => dateTime?.toIso8601String();