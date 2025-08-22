package com.arbitrationsdk.android.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize
import kotlinx.serialization.Serializable
import java.util.*
import kotlin.time.Duration

/**
 * Analysis result containing all arbitration detection information
 */
@Parcelize
@Serializable
data class ArbitrationAnalysisResult(
    val id: String,
    val hasArbitration: Boolean,
    val confidence: Double,
    val keywordMatches: List<KeywordMatch>,
    val exclusionMatches: List<KeywordMatch>,
    val riskLevel: RiskLevel,
    val recommendations: List<String>,
    val metadata: AnalysisMetadata
) : Parcelable

/**
 * Keyword match information
 */
@Parcelize
@Serializable
data class KeywordMatch(
    val keyword: String,
    val range: String, // Stored as "start:end"
    val context: String,
    val confidence: Double
) : Parcelable

/**
 * Risk level enumeration
 */
@Parcelize
@Serializable
enum class RiskLevel(val displayName: String, val colorHex: String) : Parcelable {
    MINIMAL("Minimal Risk", "#4CAF50"),
    LOW("Low Risk", "#FFC107"),
    MEDIUM("Medium Risk", "#FF9800"),
    HIGH("High Risk", "#F44336");

    companion object {
        fun fromConfidence(confidence: Double): RiskLevel {
            return when {
                confidence >= 0.8 -> HIGH
                confidence >= 0.5 -> MEDIUM
                confidence >= 0.2 -> LOW
                else -> MINIMAL
            }
        }
    }
}

/**
 * Analysis metadata
 */
@Parcelize
@Serializable
data class AnalysisMetadata(
    val textLength: Int,
    @Serializable(with = DurationSerializer::class)
    val processingTime: Duration,
    val sdkVersion: String,
    val modelVersion: String,
    @Serializable(with = DateSerializer::class)
    val analysisDate: Date
) : Parcelable

/**
 * Rule-based analysis result
 */
data class RuleBasedResult(
    val hasArbitration: Boolean,
    val confidence: Double,
    val keywordMatches: List<KeywordMatch>,
    val exclusionMatches: List<KeywordMatch>
)

/**
 * Document type enumeration
 */
@Parcelize
@Serializable
enum class DocumentType(val extensions: List<String>) : Parcelable {
    PDF(listOf("pdf")),
    WORD(listOf("doc", "docx")),
    TEXT(listOf("txt")),
    RTF(listOf("rtf")),
    HTML(listOf("html", "htm")),
    IMAGE(listOf("jpg", "jpeg", "png", "tiff", "bmp"));

    companion object {
        fun fromExtension(extension: String): DocumentType? {
            return values().find { type ->
                type.extensions.contains(extension.lowercase())
            }
        }
    }
}

/**
 * SDK Configuration
 */
@Parcelize
@Serializable
data class SDKConfiguration(
    val apiBaseUrl: String = "https://api.arbitration-platform.com",
    val apiKey: String = "",
    val confidenceThreshold: Double = 0.5,
    val enableOfflineMode: Boolean = true,
    val enableAnalytics: Boolean = true,
    val cacheSize: Int = 100,
    val networkTimeoutSeconds: Int = 30
) : Parcelable {
    companion object {
        val DEFAULT = SDKConfiguration()
    }
}

/**
 * Arbitration error types
 */
sealed class ArbitrationError(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class InvalidInput(message: String) : ArbitrationError(message)
    class AnalysisFailure(message: String, cause: Throwable? = null) : ArbitrationError(message, cause)
    class NetworkError(message: String, cause: Throwable? = null) : ArbitrationError(message, cause)
    class StorageError(message: String, cause: Throwable? = null) : ArbitrationError(message, cause)
    class UnsupportedDocumentType(message: String) : ArbitrationError(message)
    class AuthenticationFailure(message: String) : ArbitrationError(message)
    object RateLimitExceeded : ArbitrationError("Rate limit exceeded. Please try again later.")
    object ModelNotAvailable : ArbitrationError("ML model is not available. Using fallback detection.")
}

/**
 * Analysis request
 */
@Parcelize
@Serializable
data class AnalysisRequest(
    val text: String,
    val requestId: String = UUID.randomUUID().toString(),
    val options: AnalysisOptions = AnalysisOptions.DEFAULT
) : Parcelable

/**
 * Analysis options
 */
@Parcelize
@Serializable
data class AnalysisOptions(
    val includeConfidenceScore: Boolean = true,
    val includeKeywordMatches: Boolean = true,
    val includeRecommendations: Boolean = true,
    val language: String = "en",
    val strictMode: Boolean = false
) : Parcelable {
    companion object {
        val DEFAULT = AnalysisOptions()
    }
}

/**
 * Batch analysis result
 */
@Parcelize
@Serializable
data class BatchAnalysisResult(
    val batchId: String,
    val results: List<ArbitrationAnalysisResult>,
    val summary: BatchSummary,
    @Serializable(with = DurationSerializer::class)
    val processingTime: Duration
) : Parcelable

/**
 * Batch summary
 */
@Parcelize
@Serializable
data class BatchSummary(
    val totalDocuments: Int,
    val documentsWithArbitration: Int,
    val averageConfidence: Double,
    val highRiskDocuments: Int,
    val processingErrors: Int
) : Parcelable

/**
 * Analytics event
 */
@Parcelize
@Serializable
data class AnalyticsEvent(
    val eventType: AnalyticsEventType,
    @Serializable(with = DateSerializer::class)
    val timestamp: Date = Date(),
    val properties: Map<String, String> = emptyMap()
) : Parcelable

/**
 * Analytics event types
 */
@Parcelize
@Serializable
enum class AnalyticsEventType : Parcelable {
    ANALYSIS_STARTED,
    ANALYSIS_COMPLETED,
    ANALYSIS_FAILED,
    DOCUMENT_SCANNED,
    OFFLINE_MODE_ENABLED,
    CACHE_HIT,
    API_CALL_MADE
}

/**
 * Notification settings
 */
@Parcelize
@Serializable
data class NotificationSettings(
    val enablePushNotifications: Boolean = true,
    val enableAnalysisComplete: Boolean = true,
    val enableHighRiskAlerts: Boolean = true,
    val enableWeeklyReports: Boolean = false
) : Parcelable

/**
 * User preferences
 */
@Parcelize
@Serializable
data class UserPreferences(
    val language: String = "en",
    val theme: AppTheme = AppTheme.SYSTEM,
    val notifications: NotificationSettings = NotificationSettings(),
    val autoSave: Boolean = true,
    val biometricAuth: Boolean = false
) : Parcelable

/**
 * App theme
 */
@Parcelize
@Serializable
enum class AppTheme(val displayName: String) : Parcelable {
    LIGHT("Light"),
    DARK("Dark"),
    SYSTEM("System")
}

/**
 * Storage-related models
 */

/**
 * Sort options for analysis results
 */
enum class SortOption(val displayName: String) {
    DATE_ASCENDING("Date (Oldest First)"),
    DATE_DESCENDING("Date (Newest First)"),
    CONFIDENCE_ASCENDING("Confidence (Low to High)"),
    CONFIDENCE_DESCENDING("Confidence (High to Low)")
}

/**
 * Result filter
 */
data class ResultFilter(
    val hasArbitration: Boolean? = null,
    val minConfidence: Double? = null,
    val maxConfidence: Double? = null,
    val riskLevel: RiskLevel? = null,
    val dateFrom: Date? = null,
    val dateTo: Date? = null
)

/**
 * Storage statistics
 */
data class StorageStatistics(
    val totalAnalyses: Int,
    val analysesWithArbitration: Int,
    val totalEvents: Int,
    val unsyncedEvents: Int,
    val cacheHitRate: Double,
    val databaseSizeBytes: Long
)

/**
 * Network-related models
 */

/**
 * Analysis history response
 */
@Serializable
data class AnalysisHistoryResponse(
    val results: List<ArbitrationAnalysisResult>,
    val pagination: PaginationInfo,
    val totalCount: Int
)

/**
 * Pagination info
 */
@Serializable
data class PaginationInfo(
    val page: Int,
    val limit: Int,
    val hasNext: Boolean,
    val hasPrevious: Boolean
)

/**
 * API key validation response
 */
@Serializable
data class APIKeyValidationResponse(
    val isValid: Boolean,
    val permissions: List<String>,
    @Serializable(with = DateSerializer::class)
    val expiresAt: Date?,
    val rateLimit: RateLimitInfo
)

/**
 * Rate limit info
 */
@Serializable
data class RateLimitInfo(
    val requestsPerMinute: Int,
    val requestsRemaining: Int,
    @Serializable(with = DateSerializer::class)
    val resetTime: Date
)

/**
 * Usage statistics
 */
@Serializable
data class UsageStatistics(
    val totalAnalyses: Int,
    val currentMonthAnalyses: Int,
    val averageConfidence: Double,
    val topKeywords: List<String>,
    val documentTypes: Map<String, Int>,
    val successRate: Double
)

// Serializers for custom types

@kotlinx.serialization.Serializer(forClass = Duration::class)
object DurationSerializer : kotlinx.serialization.KSerializer<Duration> {
    override val descriptor = kotlinx.serialization.descriptors.PrimitiveSerialDescriptor("Duration", kotlinx.serialization.descriptors.PrimitiveKind.LONG)
    
    override fun serialize(encoder: kotlinx.serialization.encoding.Encoder, value: Duration) {
        encoder.encodeLong(value.inWholeMilliseconds)
    }
    
    override fun deserialize(decoder: kotlinx.serialization.encoding.Decoder): Duration {
        return Duration.ofMillis(decoder.decodeLong())
    }
}

@kotlinx.serialization.Serializer(forClass = Date::class)
object DateSerializer : kotlinx.serialization.KSerializer<Date> {
    override val descriptor = kotlinx.serialization.descriptors.PrimitiveSerialDescriptor("Date", kotlinx.serialization.descriptors.PrimitiveKind.LONG)
    
    override fun serialize(encoder: kotlinx.serialization.encoding.Encoder, value: Date) {
        encoder.encodeLong(value.time)
    }
    
    override fun deserialize(decoder: kotlinx.serialization.encoding.Decoder): Date {
        return Date(decoder.decodeLong())
    }
}