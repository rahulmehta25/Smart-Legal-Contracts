import Foundation
import UIKit

// MARK: - Analysis Result
public struct ArbitrationAnalysisResult: Codable, Identifiable {
    public let id: String
    public let hasArbitration: Bool
    public let confidence: Double
    public let keywordMatches: [KeywordMatch]
    public let exclusionMatches: [KeywordMatch]
    public let riskLevel: RiskLevel
    public let recommendations: [String]
    public let metadata: AnalysisMetadata
    public var processingTime: TimeInterval = 0
    
    public init(
        id: String,
        hasArbitration: Bool,
        confidence: Double,
        keywordMatches: [KeywordMatch],
        exclusionMatches: [KeywordMatch],
        riskLevel: RiskLevel,
        recommendations: [String],
        metadata: AnalysisMetadata
    ) {
        self.id = id
        self.hasArbitration = hasArbitration
        self.confidence = confidence
        self.keywordMatches = keywordMatches
        self.exclusionMatches = exclusionMatches
        self.riskLevel = riskLevel
        self.recommendations = recommendations
        self.metadata = metadata
    }
}

// MARK: - Keyword Match
public struct KeywordMatch: Codable, Identifiable {
    public let id = UUID()
    public let keyword: String
    public let range: String // Stored as string for Codable
    public let context: String
    public let confidence: Double
    
    public init(keyword: String, range: Range<String.Index>, context: String, confidence: Double) {
        self.keyword = keyword
        self.range = "\(range.lowerBound.utf16Offset(in: context)):\(range.upperBound.utf16Offset(in: context))"
        self.context = context
        self.confidence = confidence
    }
    
    private enum CodingKeys: String, CodingKey {
        case keyword, range, context, confidence
    }
}

// MARK: - Risk Level
public enum RiskLevel: String, Codable, CaseIterable {
    case minimal = "minimal"
    case low = "low"
    case medium = "medium"
    case high = "high"
    
    public var color: UIColor {
        switch self {
        case .minimal:
            return .systemGreen
        case .low:
            return .systemYellow
        case .medium:
            return .systemOrange
        case .high:
            return .systemRed
        }
    }
    
    public var description: String {
        switch self {
        case .minimal:
            return "Minimal Risk"
        case .low:
            return "Low Risk"
        case .medium:
            return "Medium Risk"
        case .high:
            return "High Risk"
        }
    }
}

// MARK: - Analysis Metadata
public struct AnalysisMetadata: Codable {
    public let textLength: Int
    public let processingTime: TimeInterval
    public let sdkVersion: String
    public let modelVersion: String
    public let analysisDate: Date
    
    public init(
        textLength: Int,
        processingTime: TimeInterval,
        sdkVersion: String,
        modelVersion: String,
        analysisDate: Date
    ) {
        self.textLength = textLength
        self.processingTime = processingTime
        self.sdkVersion = sdkVersion
        self.modelVersion = modelVersion
        self.analysisDate = analysisDate
    }
}

// MARK: - Rule Based Result
public struct RuleBasedResult {
    public let hasArbitration: Bool
    public let confidence: Double
    public let keywordMatches: [KeywordMatch]
    public let exclusionMatches: [KeywordMatch]
    
    public init(
        hasArbitration: Bool,
        confidence: Double,
        keywordMatches: [KeywordMatch],
        exclusionMatches: [KeywordMatch]
    ) {
        self.hasArbitration = hasArbitration
        self.confidence = confidence
        self.keywordMatches = keywordMatches
        self.exclusionMatches = exclusionMatches
    }
}

// MARK: - Document Type
public enum DocumentType: String, Codable, CaseIterable {
    case pdf = "pdf"
    case word = "word"
    case text = "text"
    case rtf = "rtf"
    case html = "html"
    case image = "image"
    
    public var supportedExtensions: [String] {
        switch self {
        case .pdf:
            return ["pdf"]
        case .word:
            return ["doc", "docx"]
        case .text:
            return ["txt"]
        case .rtf:
            return ["rtf"]
        case .html:
            return ["html", "htm"]
        case .image:
            return ["jpg", "jpeg", "png", "tiff", "bmp"]
        }
    }
}

// MARK: - SDK Configuration
public struct SDKConfiguration {
    public let apiBaseURL: String
    public let apiKey: String
    public let confidenceThreshold: Double
    public let enableOfflineMode: Bool
    public let enableAnalytics: Bool
    public let cacheSize: Int
    public let networkTimeout: TimeInterval
    
    public init(
        apiBaseURL: String = "https://api.arbitration-platform.com",
        apiKey: String = "",
        confidenceThreshold: Double = 0.5,
        enableOfflineMode: Bool = true,
        enableAnalytics: Bool = true,
        cacheSize: Int = 100,
        networkTimeout: TimeInterval = 30
    ) {
        self.apiBaseURL = apiBaseURL
        self.apiKey = apiKey
        self.confidenceThreshold = confidenceThreshold
        self.enableOfflineMode = enableOfflineMode
        self.enableAnalytics = enableAnalytics
        self.cacheSize = cacheSize
        self.networkTimeout = networkTimeout
    }
    
    public static let `default` = SDKConfiguration()
}

// MARK: - Arbitration Error
public enum ArbitrationError: LocalizedError {
    case invalidInput(String)
    case analysisFailure(String)
    case networkError(String)
    case storageError(String)
    case unsupportedDocumentType(String)
    case authenticationFailure(String)
    case rateLimitExceeded
    case modelNotAvailable
    
    public var errorDescription: String? {
        switch self {
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .analysisFailure(let message):
            return "Analysis failed: \(message)"
        case .networkError(let message):
            return "Network error: \(message)"
        case .storageError(let message):
            return "Storage error: \(message)"
        case .unsupportedDocumentType(let message):
            return "Unsupported document type: \(message)"
        case .authenticationFailure(let message):
            return "Authentication failed: \(message)"
        case .rateLimitExceeded:
            return "Rate limit exceeded. Please try again later."
        case .modelNotAvailable:
            return "ML model is not available. Using fallback detection."
        }
    }
}

// MARK: - Analysis Request
public struct AnalysisRequest: Codable {
    public let text: String
    public let requestId: String
    public let options: AnalysisOptions
    
    public init(text: String, requestId: String = UUID().uuidString, options: AnalysisOptions = .default) {
        self.text = text
        self.requestId = requestId
        self.options = options
    }
}

// MARK: - Analysis Options
public struct AnalysisOptions: Codable {
    public let includeConfidenceScore: Bool
    public let includeKeywordMatches: Bool
    public let includeRecommendations: Bool
    public let language: String
    public let strictMode: Bool
    
    public init(
        includeConfidenceScore: Bool = true,
        includeKeywordMatches: Bool = true,
        includeRecommendations: Bool = true,
        language: String = "en",
        strictMode: Bool = false
    ) {
        self.includeConfidenceScore = includeConfidenceScore
        self.includeKeywordMatches = includeKeywordMatches
        self.includeRecommendations = includeRecommendations
        self.language = language
        self.strictMode = strictMode
    }
    
    public static let `default` = AnalysisOptions()
}

// MARK: - Batch Analysis Result
public struct BatchAnalysisResult: Codable {
    public let batchId: String
    public let results: [ArbitrationAnalysisResult]
    public let summary: BatchSummary
    public let processingTime: TimeInterval
    
    public init(batchId: String, results: [ArbitrationAnalysisResult], summary: BatchSummary, processingTime: TimeInterval) {
        self.batchId = batchId
        self.results = results
        self.summary = summary
        self.processingTime = processingTime
    }
}

// MARK: - Batch Summary
public struct BatchSummary: Codable {
    public let totalDocuments: Int
    public let documentsWithArbitration: Int
    public let averageConfidence: Double
    public let highRiskDocuments: Int
    public let processingErrors: Int
    
    public init(
        totalDocuments: Int,
        documentsWithArbitration: Int,
        averageConfidence: Double,
        highRiskDocuments: Int,
        processingErrors: Int
    ) {
        self.totalDocuments = totalDocuments
        self.documentsWithArbitration = documentsWithArbitration
        self.averageConfidence = averageConfidence
        self.highRiskDocuments = highRiskDocuments
        self.processingErrors = processingErrors
    }
}

// MARK: - Analytics Event
public struct AnalyticsEvent: Codable {
    public let eventType: AnalyticsEventType
    public let timestamp: Date
    public let properties: [String: String]
    
    public init(eventType: AnalyticsEventType, properties: [String: String] = [:]) {
        self.eventType = eventType
        self.timestamp = Date()
        self.properties = properties
    }
}

// MARK: - Analytics Event Type
public enum AnalyticsEventType: String, Codable {
    case analysisStarted = "analysis_started"
    case analysisCompleted = "analysis_completed"
    case analysisFailed = "analysis_failed"
    case documentScanned = "document_scanned"
    case offlineModeEnabled = "offline_mode_enabled"
    case cacheHit = "cache_hit"
    case apiCallMade = "api_call_made"
}

// MARK: - Notification Settings
public struct NotificationSettings: Codable {
    public let enablePushNotifications: Bool
    public let enableAnalysisComplete: Bool
    public let enableHighRiskAlerts: Bool
    public let enableWeeklyReports: Bool
    
    public init(
        enablePushNotifications: Bool = true,
        enableAnalysisComplete: Bool = true,
        enableHighRiskAlerts: Bool = true,
        enableWeeklyReports: Bool = false
    ) {
        self.enablePushNotifications = enablePushNotifications
        self.enableAnalysisComplete = enableAnalysisComplete
        self.enableHighRiskAlerts = enableHighRiskAlerts
        self.enableWeeklyReports = enableWeeklyReports
    }
}

// MARK: - User Preferences
public struct UserPreferences: Codable {
    public let language: String
    public let theme: AppTheme
    public let notifications: NotificationSettings
    public let autoSave: Bool
    public let biometricAuth: Bool
    
    public init(
        language: String = "en",
        theme: AppTheme = .system,
        notifications: NotificationSettings = NotificationSettings(),
        autoSave: Bool = true,
        biometricAuth: Bool = false
    ) {
        self.language = language
        self.theme = theme
        self.notifications = notifications
        self.autoSave = autoSave
        self.biometricAuth = biometricAuth
    }
}

// MARK: - App Theme
public enum AppTheme: String, Codable, CaseIterable {
    case light = "light"
    case dark = "dark"
    case system = "system"
    
    public var displayName: String {
        switch self {
        case .light:
            return "Light"
        case .dark:
            return "Dark"
        case .system:
            return "System"
        }
    }
}