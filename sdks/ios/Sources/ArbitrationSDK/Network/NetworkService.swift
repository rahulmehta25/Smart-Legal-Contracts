import Foundation
import Alamofire
import Combine

/// Network service for API communication
@available(iOS 15.0, macOS 12.0, tvOS 15.0, watchOS 8.0, *)
public final class NetworkService: ObservableObject {
    
    // MARK: - Properties
    private let session: Session
    private let configuration: SDKConfiguration
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder
    private var cancellables = Set<AnyCancellable>()
    
    // Rate limiting
    private let rateLimiter = RateLimiter(requestsPerMinute: 60)
    
    // MARK: - Initialization
    public init(configuration: SDKConfiguration) {
        self.configuration = configuration
        
        // Configure URL session
        let sessionConfiguration = URLSessionConfiguration.default
        sessionConfiguration.timeoutIntervalForRequest = configuration.networkTimeout
        sessionConfiguration.timeoutIntervalForResource = configuration.networkTimeout * 2
        
        // Create Alamofire session
        self.session = Session(configuration: sessionConfiguration)
        
        // Configure JSON handling
        self.decoder = JSONDecoder()
        self.encoder = JSONEncoder()
        
        // Setup date formatting
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
        decoder.dateDecodingStrategy = .formatted(dateFormatter)
        encoder.dateEncodingStrategy = .formatted(dateFormatter)
    }
    
    // MARK: - Public Methods
    
    /// Analyze text via API
    public func analyzeText(_ request: AnalysisRequest) async throws -> ArbitrationAnalysisResult {
        guard !configuration.apiKey.isEmpty else {
            throw ArbitrationError.authenticationFailure("API key is required")
        }
        
        try await rateLimiter.waitIfNeeded()
        
        let url = "\(configuration.apiBaseURL)/v1/analyze"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "Content-Type": "application/json",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.request(
                url,
                method: .post,
                parameters: request,
                encoder: JSONParameterEncoder(encoder: encoder),
                headers: headers
            )
            .validate()
            .responseDecodable(of: ArbitrationAnalysisResult.self, decoder: decoder) { response in
                switch response.result {
                case .success(let result):
                    continuation.resume(returning: result)
                case .failure(let error):
                    let arbitrationError = self.mapAlamofireError(error, response: response.response)
                    continuation.resume(throwing: arbitrationError)
                }
            }
        }
    }
    
    /// Batch analyze multiple texts
    public func batchAnalyze(_ requests: [AnalysisRequest]) async throws -> BatchAnalysisResult {
        guard !configuration.apiKey.isEmpty else {
            throw ArbitrationError.authenticationFailure("API key is required")
        }
        
        try await rateLimiter.waitIfNeeded()
        
        let url = "\(configuration.apiBaseURL)/v1/batch-analyze"
        let batchRequest = BatchRequest(requests: requests)
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "Content-Type": "application/json",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.request(
                url,
                method: .post,
                parameters: batchRequest,
                encoder: JSONParameterEncoder(encoder: encoder),
                headers: headers
            )
            .validate()
            .responseDecodable(of: BatchAnalysisResult.self, decoder: decoder) { response in
                switch response.result {
                case .success(let result):
                    continuation.resume(returning: result)
                case .failure(let error):
                    let arbitrationError = self.mapAlamofireError(error, response: response.response)
                    continuation.resume(throwing: arbitrationError)
                }
            }
        }
    }
    
    /// Upload document for analysis
    public func uploadDocument(_ data: Data, fileName: String, contentType: String) async throws -> ArbitrationAnalysisResult {
        guard !configuration.apiKey.isEmpty else {
            throw ArbitrationError.authenticationFailure("API key is required")
        }
        
        try await rateLimiter.waitIfNeeded()
        
        let url = "\(configuration.apiBaseURL)/v1/upload-analyze"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.upload(
                multipartFormData: { multipartFormData in
                    multipartFormData.append(
                        data,
                        withName: "document",
                        fileName: fileName,
                        mimeType: contentType
                    )
                },
                to: url,
                headers: headers
            )
            .validate()
            .responseDecodable(of: ArbitrationAnalysisResult.self, decoder: decoder) { response in
                switch response.result {
                case .success(let result):
                    continuation.resume(returning: result)
                case .failure(let error):
                    let arbitrationError = self.mapAlamofireError(error, response: response.response)
                    continuation.resume(throwing: arbitrationError)
                }
            }
        }
    }
    
    /// Get analysis history
    public func getAnalysisHistory(page: Int = 1, limit: Int = 20) async throws -> AnalysisHistoryResponse {
        guard !configuration.apiKey.isEmpty else {
            throw ArbitrationError.authenticationFailure("API key is required")
        }
        
        let url = "\(configuration.apiBaseURL)/v1/history"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        let parameters: [String: Any] = [
            "page": page,
            "limit": limit
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.request(
                url,
                method: .get,
                parameters: parameters,
                headers: headers
            )
            .validate()
            .responseDecodable(of: AnalysisHistoryResponse.self, decoder: decoder) { response in
                switch response.result {
                case .success(let result):
                    continuation.resume(returning: result)
                case .failure(let error):
                    let arbitrationError = self.mapAlamofireError(error, response: response.response)
                    continuation.resume(throwing: arbitrationError)
                }
            }
        }
    }
    
    /// Validate API key
    public func validateAPIKey() async throws -> APIKeyValidationResponse {
        let url = "\(configuration.apiBaseURL)/v1/validate-key"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.request(url, headers: headers)
                .validate()
                .responseDecodable(of: APIKeyValidationResponse.self, decoder: decoder) { response in
                    switch response.result {
                    case .success(let result):
                        continuation.resume(returning: result)
                    case .failure(let error):
                        let arbitrationError = self.mapAlamofireError(error, response: response.response)
                        continuation.resume(throwing: arbitrationError)
                    }
                }
        }
    }
    
    /// Get SDK usage statistics
    public func getUsageStatistics() async throws -> UsageStatistics {
        guard !configuration.apiKey.isEmpty else {
            throw ArbitrationError.authenticationFailure("API key is required")
        }
        
        let url = "\(configuration.apiBaseURL)/v1/usage"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        return try await withCheckedThrowingContinuation { continuation in
            session.request(url, headers: headers)
                .validate()
                .responseDecodable(of: UsageStatistics.self, decoder: decoder) { response in
                    switch response.result {
                    case .success(let result):
                        continuation.resume(returning: result)
                    case .failure(let error):
                        let arbitrationError = self.mapAlamofireError(error, response: response.response)
                        continuation.resume(throwing: arbitrationError)
                    }
                }
        }
    }
    
    /// Send analytics events
    public func sendAnalyticsEvent(_ event: AnalyticsEvent) async throws {
        guard configuration.enableAnalytics else { return }
        
        let url = "\(configuration.apiBaseURL)/v1/analytics"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer \(configuration.apiKey)",
            "Content-Type": "application/json",
            "User-Agent": "ArbitrationSDK-iOS/1.0.0"
        ]
        
        _ = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            session.request(
                url,
                method: .post,
                parameters: event,
                encoder: JSONParameterEncoder(encoder: encoder),
                headers: headers
            )
            .validate()
            .response { response in
                switch response.result {
                case .success:
                    continuation.resume(returning: ())
                case .failure(let error):
                    // Analytics errors shouldn't fail the main operation
                    print("Analytics error: \(error)")
                    continuation.resume(returning: ())
                }
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func mapAlamofireError(_ error: AFError, response: HTTPURLResponse?) -> ArbitrationError {
        switch error {
        case .responseValidationFailed(reason: .unacceptableStatusCode(let code)):
            switch code {
            case 401:
                return .authenticationFailure("Invalid API key")
            case 429:
                return .rateLimitExceeded
            case 500...599:
                return .networkError("Server error (HTTP \(code))")
            default:
                return .networkError("HTTP error \(code)")
            }
        case .sessionTaskFailed(let urlError as URLError):
            switch urlError.code {
            case .notConnectedToInternet, .networkConnectionLost:
                return .networkError("No internet connection")
            case .timedOut:
                return .networkError("Request timed out")
            default:
                return .networkError(urlError.localizedDescription)
            }
        default:
            return .networkError(error.localizedDescription)
        }
    }
}

// MARK: - Rate Limiter
private actor RateLimiter {
    private let requestsPerMinute: Int
    private var requests: [Date] = []
    
    init(requestsPerMinute: Int) {
        self.requestsPerMinute = requestsPerMinute
    }
    
    func waitIfNeeded() async throws {
        let now = Date()
        let oneMinuteAgo = now.addingTimeInterval(-60)
        
        // Remove old requests
        requests = requests.filter { $0 > oneMinuteAgo }
        
        if requests.count >= requestsPerMinute {
            // Calculate wait time
            let oldestRequest = requests.first ?? now
            let waitTime = 60 - now.timeIntervalSince(oldestRequest)
            
            if waitTime > 0 {
                try await Task.sleep(nanoseconds: UInt64(waitTime * 1_000_000_000))
            }
        }
        
        requests.append(now)
    }
}

// MARK: - Network Models
private struct BatchRequest: Codable {
    let requests: [AnalysisRequest]
}

public struct AnalysisHistoryResponse: Codable {
    public let results: [ArbitrationAnalysisResult]
    public let pagination: PaginationInfo
    public let totalCount: Int
}

public struct PaginationInfo: Codable {
    public let page: Int
    public let limit: Int
    public let hasNext: Bool
    public let hasPrevious: Bool
}

public struct APIKeyValidationResponse: Codable {
    public let isValid: Bool
    public let permissions: [String]
    public let expiresAt: Date?
    public let rateLimit: RateLimitInfo
}

public struct RateLimitInfo: Codable {
    public let requestsPerMinute: Int
    public let requestsRemaining: Int
    public let resetTime: Date
}

public struct UsageStatistics: Codable {
    public let totalAnalyses: Int
    public let currentMonthAnalyses: Int
    public let averageConfidence: Double
    public let topKeywords: [String]
    public let documentTypes: [String: Int]
    public let successRate: Double
}