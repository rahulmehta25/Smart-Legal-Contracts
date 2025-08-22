import Foundation
import CoreML
import NaturalLanguage
import Vision
import Combine

/// Main arbitration detection engine for iOS
@available(iOS 15.0, macOS 12.0, tvOS 15.0, watchOS 8.0, *)
public final class ArbitrationDetector: ObservableObject {
    
    // MARK: - Published Properties
    @Published public private(set) var isAnalyzing = false
    @Published public private(set) var analysisProgress: Double = 0.0
    @Published public private(set) var lastAnalysisResult: ArbitrationAnalysisResult?
    
    // MARK: - Private Properties
    private let mlModel: MLModel?
    private let networkService: NetworkService
    private let storageService: StorageService
    private let configuration: SDKConfiguration
    private var analysisTask: Task<Void, Error>?
    private let analysisQueue = DispatchQueue(label: "com.arbitrationsdk.analysis", qos: .userInitiated)
    
    // MARK: - Initialization
    public init(configuration: SDKConfiguration = .default) {
        self.configuration = configuration
        self.networkService = NetworkService(configuration: configuration)
        self.storageService = StorageService()
        
        // Load Core ML model
        do {
            if let modelURL = Bundle.module.url(forResource: "ArbitrationDetectionModel", withExtension: "mlmodelc") {
                self.mlModel = try MLModel(contentsOf: modelURL)
            } else {
                self.mlModel = nil
                print("Warning: Core ML model not found. Using fallback detection.")
            }
        } catch {
            self.mlModel = nil
            print("Error loading Core ML model: \(error)")
        }
    }
    
    // MARK: - Public Methods
    
    /// Analyze text for arbitration clauses
    public func analyzeText(_ text: String) async throws -> ArbitrationAnalysisResult {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ArbitrationError.invalidInput("Text cannot be empty")
        }
        
        await MainActor.run {
            isAnalyzing = true
            analysisProgress = 0.0
        }
        
        defer {
            Task { @MainActor in
                isAnalyzing = false
                analysisProgress = 0.0
            }
        }
        
        do {
            let result = try await performAnalysis(text: text)
            
            // Store result locally
            try await storageService.saveAnalysisResult(result)
            
            await MainActor.run {
                lastAnalysisResult = result
            }
            
            return result
        } catch {
            throw ArbitrationError.analysisFailure(error.localizedDescription)
        }
    }
    
    /// Analyze document from URL or file path
    public func analyzeDocument(_ documentData: Data, type: DocumentType) async throws -> ArbitrationAnalysisResult {
        let extractedText = try await extractTextFromDocument(documentData, type: type)
        return try await analyzeText(extractedText)
    }
    
    /// Analyze image containing text
    public func analyzeImage(_ image: UIImage) async throws -> ArbitrationAnalysisResult {
        let extractedText = try await extractTextFromImage(image)
        return try await analyzeText(extractedText)
    }
    
    /// Cancel ongoing analysis
    public func cancelAnalysis() {
        analysisTask?.cancel()
        Task { @MainActor in
            isAnalyzing = false
            analysisProgress = 0.0
        }
    }
    
    // MARK: - Private Methods
    
    private func performAnalysis(text: String) async throws -> ArbitrationAnalysisResult {
        let analysisId = UUID().uuidString
        let startTime = Date()
        
        await updateProgress(0.1)
        
        // Step 1: Preprocessing
        let preprocessedText = preprocessText(text)
        await updateProgress(0.2)
        
        // Step 2: Feature extraction
        let features = try await extractFeatures(from: preprocessedText)
        await updateProgress(0.4)
        
        // Step 3: ML Model prediction (if available)
        var mlConfidence: Double = 0.0
        var mlResult: Bool = false
        
        if let model = mlModel {
            (mlResult, mlConfidence) = try await predictWithMLModel(model, features: features)
        }
        await updateProgress(0.6)
        
        // Step 4: Rule-based analysis
        let ruleBasedResult = performRuleBasedAnalysis(preprocessedText)
        await updateProgress(0.8)
        
        // Step 5: Combine results
        let finalResult = combineAnalysisResults(
            mlResult: mlResult,
            mlConfidence: mlConfidence,
            ruleBasedResult: ruleBasedResult,
            originalText: text,
            analysisId: analysisId
        )
        
        await updateProgress(1.0)
        
        let endTime = Date()
        finalResult.processingTime = endTime.timeIntervalSince(startTime)
        
        return finalResult
    }
    
    private func preprocessText(_ text: String) -> String {
        return text
            .lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func extractFeatures(from text: String) async throws -> [String: Any] {
        return try await withCheckedThrowingContinuation { continuation in
            analysisQueue.async {
                do {
                    let features: [String: Any] = [
                        "text_length": text.count,
                        "word_count": text.components(separatedBy: .whitespacesAndNewlines).count,
                        "arbitration_keywords": self.countArbitrationKeywords(in: text),
                        "legal_terms": self.countLegalTerms(in: text),
                        "sentiment_score": self.analyzeSentiment(text),
                        "complexity_score": self.calculateComplexityScore(text)
                    ]
                    continuation.resume(returning: features)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    private func predictWithMLModel(_ model: MLModel, features: [String: Any]) async throws -> (Bool, Double) {
        return try await withCheckedThrowingContinuation { continuation in
            do {
                // Convert features to ML input format
                let input = try MLDictionaryFeatureProvider(dictionary: features)
                let prediction = try model.prediction(from: input)
                
                // Extract prediction results
                if let probabilityDict = prediction.featureValue(for: "probability")?.dictionaryValue,
                   let arbitrationProbability = probabilityDict["arbitration"]?.doubleValue,
                   let classLabel = prediction.featureValue(for: "class")?.stringValue {
                    
                    let hasArbitration = classLabel == "arbitration"
                    continuation.resume(returning: (hasArbitration, arbitrationProbability))
                } else {
                    continuation.resume(returning: (false, 0.0))
                }
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    private func performRuleBasedAnalysis(_ text: String) -> RuleBasedResult {
        let arbitrationKeywords = [
            "arbitration", "arbitrator", "arbitral", "binding arbitration",
            "alternative dispute resolution", "adr", "arbitration clause",
            "dispute resolution", "binding resolution", "final and binding"
        ]
        
        let exclusionPhrases = [
            "court proceedings", "litigation", "judicial review",
            "class action", "jury trial", "court of law"
        ]
        
        var keywordMatches: [KeywordMatch] = []
        var exclusionMatches: [KeywordMatch] = []
        
        // Find arbitration keywords
        for keyword in arbitrationKeywords {
            let ranges = text.ranges(of: keyword, options: .caseInsensitive)
            for range in ranges {
                let match = KeywordMatch(
                    keyword: keyword,
                    range: range,
                    context: extractContext(from: text, around: range),
                    confidence: calculateKeywordConfidence(keyword)
                )
                keywordMatches.append(match)
            }
        }
        
        // Find exclusion phrases
        for phrase in exclusionPhrases {
            let ranges = text.ranges(of: phrase, options: .caseInsensitive)
            for range in ranges {
                let match = KeywordMatch(
                    keyword: phrase,
                    range: range,
                    context: extractContext(from: text, around: range),
                    confidence: 0.8
                )
                exclusionMatches.append(match)
            }
        }
        
        let confidence = calculateOverallConfidence(
            keywordMatches: keywordMatches,
            exclusionMatches: exclusionMatches
        )
        
        return RuleBasedResult(
            hasArbitration: !keywordMatches.isEmpty && confidence > 0.5,
            confidence: confidence,
            keywordMatches: keywordMatches,
            exclusionMatches: exclusionMatches
        )
    }
    
    private func combineAnalysisResults(
        mlResult: Bool,
        mlConfidence: Double,
        ruleBasedResult: RuleBasedResult,
        originalText: String,
        analysisId: String
    ) -> ArbitrationAnalysisResult {
        
        // Weighted combination of ML and rule-based results
        let mlWeight = 0.7
        let ruleWeight = 0.3
        
        let combinedConfidence = (mlConfidence * mlWeight) + (ruleBasedResult.confidence * ruleWeight)
        let hasArbitration = combinedConfidence > configuration.confidenceThreshold
        
        return ArbitrationAnalysisResult(
            id: analysisId,
            hasArbitration: hasArbitration,
            confidence: combinedConfidence,
            keywordMatches: ruleBasedResult.keywordMatches,
            exclusionMatches: ruleBasedResult.exclusionMatches,
            riskLevel: calculateRiskLevel(confidence: combinedConfidence),
            recommendations: generateRecommendations(
                hasArbitration: hasArbitration,
                confidence: combinedConfidence,
                keywordMatches: ruleBasedResult.keywordMatches
            ),
            metadata: AnalysisMetadata(
                textLength: originalText.count,
                processingTime: 0, // Will be set later
                sdkVersion: "1.0.0",
                modelVersion: "1.0.0",
                analysisDate: Date()
            )
        )
    }
    
    // MARK: - Helper Methods
    
    private func countArbitrationKeywords(in text: String) -> Int {
        let keywords = ["arbitration", "arbitrator", "arbitral", "adr"]
        return keywords.reduce(0) { count, keyword in
            count + text.components(separatedBy: keyword).count - 1
        }
    }
    
    private func countLegalTerms(in text: String) -> Int {
        let terms = ["whereas", "hereby", "parties", "agreement", "contract"]
        return terms.reduce(0) { count, term in
            count + text.components(separatedBy: term).count - 1
        }
    }
    
    private func analyzeSentiment(_ text: String) -> Double {
        let tagger = NLTagger(tagSchemes: [.sentimentScore])
        tagger.string = text
        
        let sentiment = tagger.tag(at: text.startIndex, unit: .paragraph, scheme: .sentimentScore).0
        return Double(sentiment?.rawValue ?? "0") ?? 0.0
    }
    
    private func calculateComplexityScore(_ text: String) -> Double {
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        let avgWordLength = words.map { $0.count }.reduce(0, +) / max(words.count, 1)
        let sentences = text.components(separatedBy: .punctuationCharacters)
        let avgSentenceLength = words.count / max(sentences.count, 1)
        
        return Double(avgWordLength + avgSentenceLength) / 10.0
    }
    
    private func calculateKeywordConfidence(_ keyword: String) -> Double {
        let weights: [String: Double] = [
            "arbitration": 0.9,
            "arbitrator": 0.8,
            "arbitral": 0.8,
            "binding arbitration": 0.95,
            "alternative dispute resolution": 0.7,
            "adr": 0.6
        ]
        return weights[keyword.lowercased()] ?? 0.5
    }
    
    private func extractContext(from text: String, around range: Range<String.Index>) -> String {
        let contextLength = 50
        let startIndex = max(text.startIndex, text.index(range.lowerBound, offsetBy: -contextLength, limitedBy: text.startIndex) ?? text.startIndex)
        let endIndex = min(text.endIndex, text.index(range.upperBound, offsetBy: contextLength, limitedBy: text.endIndex) ?? text.endIndex)
        return String(text[startIndex..<endIndex])
    }
    
    private func calculateOverallConfidence(keywordMatches: [KeywordMatch], exclusionMatches: [KeywordMatch]) -> Double {
        guard !keywordMatches.isEmpty else { return 0.0 }
        
        let keywordConfidence = keywordMatches.map { $0.confidence }.reduce(0, +) / Double(keywordMatches.count)
        let exclusionPenalty = exclusionMatches.isEmpty ? 0.0 : 0.3
        
        return max(0.0, min(1.0, keywordConfidence - exclusionPenalty))
    }
    
    private func calculateRiskLevel(confidence: Double) -> RiskLevel {
        switch confidence {
        case 0.8...:
            return .high
        case 0.5..<0.8:
            return .medium
        case 0.2..<0.5:
            return .low
        default:
            return .minimal
        }
    }
    
    private func generateRecommendations(hasArbitration: Bool, confidence: Double, keywordMatches: [KeywordMatch]) -> [String] {
        var recommendations: [String] = []
        
        if hasArbitration {
            recommendations.append("Arbitration clause detected. Review the specific terms and conditions.")
            
            if confidence > 0.8 {
                recommendations.append("High confidence detection. Consider legal consultation for detailed analysis.")
            } else {
                recommendations.append("Moderate confidence detection. Manual review recommended.")
            }
            
            if keywordMatches.contains(where: { $0.keyword.contains("binding") }) {
                recommendations.append("Binding arbitration clause found. This may limit legal recourse options.")
            }
        } else {
            recommendations.append("No clear arbitration clause detected in the analyzed text.")
            
            if confidence < 0.2 {
                recommendations.append("Consider reviewing the full document context for complete analysis.")
            }
        }
        
        return recommendations
    }
    
    private func extractTextFromDocument(_ data: Data, type: DocumentType) async throws -> String {
        // Implementation would depend on document type
        // For now, returning a placeholder
        throw ArbitrationError.unsupportedDocumentType("Document text extraction not yet implemented")
    }
    
    private func extractTextFromImage(_ image: UIImage) async throws -> String {
        return try await withCheckedThrowingContinuation { continuation in
            guard let cgImage = image.cgImage else {
                continuation.resume(throwing: ArbitrationError.invalidInput("Invalid image"))
                return
            }
            
            let request = VNRecognizeTextRequest { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                let observations = request.results as? [VNRecognizedTextObservation] ?? []
                let recognizedText = observations.compactMap { observation in
                    observation.topCandidates(1).first?.string
                }.joined(separator: " ")
                
                continuation.resume(returning: recognizedText)
            }
            
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = true
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
    
    @MainActor
    private func updateProgress(_ progress: Double) {
        analysisProgress = progress
    }
}

// MARK: - String Extension
private extension String {
    func ranges(of substring: String, options: CompareOptions = []) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []
        var searchStartIndex = self.startIndex
        
        while searchStartIndex < self.endIndex,
              let range = self.range(of: substring, options: options, range: searchStartIndex..<self.endIndex) {
            ranges.append(range)
            searchStartIndex = range.upperBound
        }
        
        return ranges
    }
}