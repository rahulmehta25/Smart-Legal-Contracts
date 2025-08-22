package com.arbitrationsdk.android.core

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.arbitrationsdk.android.ml.MLModelManager
import com.arbitrationsdk.android.network.NetworkService
import com.arbitrationsdk.android.storage.StorageService
import com.arbitrationsdk.android.models.*
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber
import java.util.*
import javax.inject.Inject
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

/**
 * Main arbitration detection engine for Android
 * Provides comprehensive arbitration clause detection with ML and rule-based analysis
 */
@HiltViewModel
class ArbitrationDetector @Inject constructor(
    @ApplicationContext private val context: Context,
    private val networkService: NetworkService,
    private val storageService: StorageService,
    private val mlModelManager: MLModelManager,
    private val configuration: SDKConfiguration
) : ViewModel() {

    // State flows for reactive UI
    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val _analysisProgress = MutableStateFlow(0f)
    val analysisProgress: StateFlow<Float> = _analysisProgress.asStateFlow()

    private val _lastAnalysisResult = MutableStateFlow<ArbitrationAnalysisResult?>(null)
    val lastAnalysisResult: StateFlow<ArbitrationAnalysisResult?> = _lastAnalysisResult.asStateFlow()

    private val _error = MutableSharedFlow<ArbitrationError>()
    val error: SharedFlow<ArbitrationError> = _error.asSharedFlow()

    // Analysis state
    private var currentAnalysisJob: kotlinx.coroutines.Job? = null

    init {
        // Initialize ML models
        viewModelScope.launch {
            try {
                mlModelManager.initializeModels()
                Timber.d("ML models initialized successfully")
            } catch (e: Exception) {
                Timber.e(e, "Failed to initialize ML models")
            }
        }
    }

    /**
     * Analyze text for arbitration clauses
     */
    suspend fun analyzeText(text: String): Result<ArbitrationAnalysisResult> {
        if (text.trim().isEmpty()) {
            return Result.failure(ArbitrationError.InvalidInput("Text cannot be empty"))
        }

        return withContext(Dispatchers.IO) {
            try {
                _isAnalyzing.value = true
                _analysisProgress.value = 0f

                val result = performAnalysis(text)
                
                // Store result locally
                storageService.saveAnalysisResult(result)
                
                _lastAnalysisResult.value = result
                Result.success(result)
            } catch (e: Exception) {
                val error = when (e) {
                    is ArbitrationError -> e
                    else -> ArbitrationError.AnalysisFailure(e.message ?: "Unknown error")
                }
                _error.emit(error)
                Result.failure(error)
            } finally {
                _isAnalyzing.value = false
                _analysisProgress.value = 0f
            }
        }
    }

    /**
     * Analyze document from byte array
     */
    suspend fun analyzeDocument(
        data: ByteArray,
        documentType: DocumentType
    ): Result<ArbitrationAnalysisResult> {
        return withContext(Dispatchers.IO) {
            try {
                val extractedText = extractTextFromDocument(data, documentType)
                analyzeText(extractedText)
            } catch (e: Exception) {
                Result.failure(ArbitrationError.UnsupportedDocumentType("Failed to extract text: ${e.message}"))
            }
        }
    }

    /**
     * Analyze image with OCR
     */
    suspend fun analyzeImage(imageData: ByteArray): Result<ArbitrationAnalysisResult> {
        return withContext(Dispatchers.IO) {
            try {
                val extractedText = extractTextFromImage(imageData)
                analyzeText(extractedText)
            } catch (e: Exception) {
                Result.failure(ArbitrationError.AnalysisFailure("OCR failed: ${e.message}"))
            }
        }
    }

    /**
     * Batch analyze multiple texts
     */
    suspend fun batchAnalyze(
        requests: List<AnalysisRequest>
    ): Result<BatchAnalysisResult> {
        if (requests.isEmpty()) {
            return Result.failure(ArbitrationError.InvalidInput("No requests provided"))
        }

        return withContext(Dispatchers.IO) {
            try {
                _isAnalyzing.value = true
                val batchId = UUID.randomUUID().toString()
                val results = mutableListOf<ArbitrationAnalysisResult>()
                val startTime = System.currentTimeMillis()

                requests.forEachIndexed { index, request ->
                    _analysisProgress.value = index.toFloat() / requests.size
                    
                    val result = performAnalysis(request.text)
                    results.add(result)
                    
                    // Store individual results
                    storageService.saveAnalysisResult(result)
                }

                val endTime = System.currentTimeMillis()
                val processingTime = (endTime - startTime).toDuration(DurationUnit.MILLISECONDS)

                val summary = BatchSummary(
                    totalDocuments = results.size,
                    documentsWithArbitration = results.count { it.hasArbitration },
                    averageConfidence = results.map { it.confidence }.average(),
                    highRiskDocuments = results.count { it.riskLevel == RiskLevel.HIGH },
                    processingErrors = 0
                )

                val batchResult = BatchAnalysisResult(
                    batchId = batchId,
                    results = results,
                    summary = summary,
                    processingTime = processingTime
                )

                Result.success(batchResult)
            } catch (e: Exception) {
                Result.failure(ArbitrationError.AnalysisFailure(e.message ?: "Batch analysis failed"))
            } finally {
                _isAnalyzing.value = false
                _analysisProgress.value = 0f
            }
        }
    }

    /**
     * Cancel ongoing analysis
     */
    fun cancelAnalysis() {
        currentAnalysisJob?.cancel()
        _isAnalyzing.value = false
        _analysisProgress.value = 0f
    }

    /**
     * Get analysis history
     */
    fun getAnalysisHistory(
        limit: Int = 50,
        offset: Int = 0,
        sortBy: SortOption = SortOption.DATE_DESCENDING,
        filter: ResultFilter? = null
    ): Flow<List<ArbitrationAnalysisResult>> {
        return storageService.getAllAnalysisResults(limit, offset, sortBy, filter)
    }

    // Private implementation methods

    private suspend fun performAnalysis(text: String): ArbitrationAnalysisResult {
        val analysisId = UUID.randomUUID().toString()
        val startTime = System.currentTimeMillis()

        updateProgress(0.1f)

        // Step 1: Preprocessing
        val preprocessedText = preprocessText(text)
        updateProgress(0.2f)

        // Step 2: Feature extraction
        val features = extractFeatures(preprocessedText)
        updateProgress(0.4f)

        // Step 3: ML Model prediction
        val (mlResult, mlConfidence) = if (mlModelManager.isModelAvailable()) {
            mlModelManager.predict(features)
        } else {
            Pair(false, 0.0)
        }
        updateProgress(0.6f)

        // Step 4: Rule-based analysis
        val ruleBasedResult = performRuleBasedAnalysis(preprocessedText)
        updateProgress(0.8f)

        // Step 5: Combine results
        val finalResult = combineAnalysisResults(
            mlResult = mlResult,
            mlConfidence = mlConfidence,
            ruleBasedResult = ruleBasedResult,
            originalText = text,
            analysisId = analysisId
        )

        updateProgress(1.0f)

        val endTime = System.currentTimeMillis()
        val processingTime = (endTime - startTime).toDuration(DurationUnit.MILLISECONDS)
        
        return finalResult.copy(
            metadata = finalResult.metadata.copy(
                processingTime = processingTime
            )
        )
    }

    private fun preprocessText(text: String): String {
        return text.lowercase(Locale.getDefault())
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun extractFeatures(text: String): Map<String, Any> {
        val words = text.split(Regex("\\s+"))
        
        return mapOf(
            "text_length" to text.length,
            "word_count" to words.size,
            "arbitration_keywords" to countArbitrationKeywords(text),
            "legal_terms" to countLegalTerms(text),
            "sentence_count" to text.split(Regex("[.!?]+")).size,
            "avg_word_length" to if (words.isNotEmpty()) words.map { it.length }.average() else 0.0,
            "complexity_score" to calculateComplexityScore(text)
        )
    }

    private fun countArbitrationKeywords(text: String): Int {
        val keywords = listOf(
            "arbitration", "arbitrator", "arbitral", "adr",
            "binding arbitration", "alternative dispute resolution"
        )
        return keywords.sumOf { keyword ->
            text.split(keyword, ignoreCase = true).size - 1
        }
    }

    private fun countLegalTerms(text: String): Int {
        val terms = listOf(
            "whereas", "hereby", "parties", "agreement", "contract",
            "terms", "conditions", "jurisdiction", "governed"
        )
        return terms.sumOf { term ->
            text.split(term, ignoreCase = true).size - 1
        }
    }

    private fun calculateComplexityScore(text: String): Double {
        val words = text.split(Regex("\\s+"))
        val avgWordLength = if (words.isNotEmpty()) words.map { it.length }.average() else 0.0
        val sentences = text.split(Regex("[.!?]+"))
        val avgSentenceLength = if (sentences.isNotEmpty()) words.size.toDouble() / sentences.size else 0.0
        
        return (avgWordLength + avgSentenceLength) / 10.0
    }

    private fun performRuleBasedAnalysis(text: String): RuleBasedResult {
        val arbitrationKeywords = listOf(
            "arbitration", "arbitrator", "arbitral", "binding arbitration",
            "alternative dispute resolution", "adr", "arbitration clause",
            "dispute resolution", "binding resolution", "final and binding"
        )

        val exclusionPhrases = listOf(
            "court proceedings", "litigation", "judicial review",
            "class action", "jury trial", "court of law"
        )

        val keywordMatches = mutableListOf<KeywordMatch>()
        val exclusionMatches = mutableListOf<KeywordMatch>()

        // Find arbitration keywords
        arbitrationKeywords.forEach { keyword ->
            val regex = Regex("\\b${Regex.escape(keyword)}\\b", RegexOption.IGNORE_CASE)
            regex.findAll(text).forEach { match ->
                val keywordMatch = KeywordMatch(
                    keyword = keyword,
                    range = "${match.range.first}:${match.range.last}",
                    context = extractContext(text, match.range),
                    confidence = calculateKeywordConfidence(keyword)
                )
                keywordMatches.add(keywordMatch)
            }
        }

        // Find exclusion phrases
        exclusionPhrases.forEach { phrase ->
            val regex = Regex("\\b${Regex.escape(phrase)}\\b", RegexOption.IGNORE_CASE)
            regex.findAll(text).forEach { match ->
                val exclusionMatch = KeywordMatch(
                    keyword = phrase,
                    range = "${match.range.first}:${match.range.last}",
                    context = extractContext(text, match.range),
                    confidence = 0.8
                )
                exclusionMatches.add(exclusionMatch)
            }
        }

        val confidence = calculateOverallConfidence(keywordMatches, exclusionMatches)

        return RuleBasedResult(
            hasArbitration = keywordMatches.isNotEmpty() && confidence > 0.5,
            confidence = confidence,
            keywordMatches = keywordMatches,
            exclusionMatches = exclusionMatches
        )
    }

    private fun extractContext(text: String, range: IntRange): String {
        val contextLength = 50
        val start = maxOf(0, range.first - contextLength)
        val end = minOf(text.length, range.last + contextLength)
        return text.substring(start, end)
    }

    private fun calculateKeywordConfidence(keyword: String): Double {
        val weights = mapOf(
            "arbitration" to 0.9,
            "arbitrator" to 0.8,
            "arbitral" to 0.8,
            "binding arbitration" to 0.95,
            "alternative dispute resolution" to 0.7,
            "adr" to 0.6
        )
        return weights[keyword.lowercase()] ?: 0.5
    }

    private fun calculateOverallConfidence(
        keywordMatches: List<KeywordMatch>,
        exclusionMatches: List<KeywordMatch>
    ): Double {
        if (keywordMatches.isEmpty()) return 0.0

        val keywordConfidence = keywordMatches.map { it.confidence }.average()
        val exclusionPenalty = if (exclusionMatches.isNotEmpty()) 0.3 else 0.0

        return maxOf(0.0, minOf(1.0, keywordConfidence - exclusionPenalty))
    }

    private fun combineAnalysisResults(
        mlResult: Boolean,
        mlConfidence: Double,
        ruleBasedResult: RuleBasedResult,
        originalText: String,
        analysisId: String
    ): ArbitrationAnalysisResult {
        // Weighted combination of ML and rule-based results
        val mlWeight = 0.7
        val ruleWeight = 0.3

        val combinedConfidence = (mlConfidence * mlWeight) + (ruleBasedResult.confidence * ruleWeight)
        val hasArbitration = combinedConfidence > configuration.confidenceThreshold

        return ArbitrationAnalysisResult(
            id = analysisId,
            hasArbitration = hasArbitration,
            confidence = combinedConfidence,
            keywordMatches = ruleBasedResult.keywordMatches,
            exclusionMatches = ruleBasedResult.exclusionMatches,
            riskLevel = calculateRiskLevel(combinedConfidence),
            recommendations = generateRecommendations(hasArbitration, combinedConfidence, ruleBasedResult.keywordMatches),
            metadata = AnalysisMetadata(
                textLength = originalText.length,
                processingTime = Duration.ZERO, // Will be set later
                sdkVersion = "1.0.0",
                modelVersion = "1.0.0",
                analysisDate = Date()
            )
        )
    }

    private fun calculateRiskLevel(confidence: Double): RiskLevel {
        return when {
            confidence >= 0.8 -> RiskLevel.HIGH
            confidence >= 0.5 -> RiskLevel.MEDIUM
            confidence >= 0.2 -> RiskLevel.LOW
            else -> RiskLevel.MINIMAL
        }
    }

    private fun generateRecommendations(
        hasArbitration: Boolean,
        confidence: Double,
        keywordMatches: List<KeywordMatch>
    ): List<String> {
        val recommendations = mutableListOf<String>()

        if (hasArbitration) {
            recommendations.add("Arbitration clause detected. Review the specific terms and conditions.")

            if (confidence > 0.8) {
                recommendations.add("High confidence detection. Consider legal consultation for detailed analysis.")
            } else {
                recommendations.add("Moderate confidence detection. Manual review recommended.")
            }

            if (keywordMatches.any { it.keyword.contains("binding", ignoreCase = true) }) {
                recommendations.add("Binding arbitration clause found. This may limit legal recourse options.")
            }
        } else {
            recommendations.add("No clear arbitration clause detected in the analyzed text.")

            if (confidence < 0.2) {
                recommendations.add("Consider reviewing the full document context for complete analysis.")
            }
        }

        return recommendations
    }

    private suspend fun extractTextFromDocument(data: ByteArray, type: DocumentType): String {
        // Placeholder for document text extraction
        // In a real implementation, you would use libraries like Apache Tika or specific parsers
        throw ArbitrationError.UnsupportedDocumentType("Document text extraction not yet implemented")
    }

    private suspend fun extractTextFromImage(imageData: ByteArray): String {
        // Placeholder for OCR implementation
        // In a real implementation, you would use ML Kit Text Recognition or Tesseract
        throw ArbitrationError.AnalysisFailure("OCR text extraction not yet implemented")
    }

    private suspend fun updateProgress(progress: Float) {
        _analysisProgress.value = progress
    }

    override fun onCleared() {
        super.onCleared()
        cancelAnalysis()
    }
}