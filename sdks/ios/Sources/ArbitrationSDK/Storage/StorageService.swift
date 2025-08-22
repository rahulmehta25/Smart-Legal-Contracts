import Foundation
import CoreData
import Combine

/// Storage service for local data persistence
@available(iOS 15.0, macOS 12.0, tvOS 15.0, watchOS 8.0, *)
public final class StorageService: ObservableObject {
    
    // MARK: - Properties
    private let container: NSPersistentContainer
    private let context: NSManagedObjectContext
    private var cancellables = Set<AnyCancellable>()
    
    // Cache for analysis results
    private let cache = NSCache<NSString, ArbitrationAnalysisResult>()
    private let cacheQueue = DispatchQueue(label: "com.arbitrationsdk.cache", qos: .utility)
    
    // MARK: - Initialization
    public init() {
        // Create Core Data stack
        container = NSPersistentContainer(name: "ArbitrationDataModel")
        
        // Configure persistent store
        let storeDescription = container.persistentStoreDescriptions.first!
        storeDescription.setOption(true as NSNumber, forKey: NSPersistentHistoryTrackingKey)
        storeDescription.setOption(true as NSNumber, forKey: NSPersistentStoreRemoteChangeNotificationPostOptionKey)
        
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Core Data error: \(error)")
            }
        }
        
        context = container.viewContext
        context.automaticallyMergesChangesFromParent = true
        
        // Configure cache
        cache.countLimit = 100
        cache.totalCostLimit = 50 * 1024 * 1024 // 50MB
        
        // Setup remote change notifications
        setupRemoteChangeNotifications()
    }
    
    // MARK: - Analysis Results
    
    /// Save analysis result to local storage
    public func saveAnalysisResult(_ result: ArbitrationAnalysisResult) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    // Check if result already exists
                    let fetchRequest: NSFetchRequest<AnalysisResultEntity> = AnalysisResultEntity.fetchRequest()
                    fetchRequest.predicate = NSPredicate(format: "id == %@", result.id)
                    
                    let existingResults = try self.context.fetch(fetchRequest)
                    let entity = existingResults.first ?? AnalysisResultEntity(context: self.context)
                    
                    // Update entity
                    entity.id = result.id
                    entity.hasArbitration = result.hasArbitration
                    entity.confidence = result.confidence
                    entity.riskLevel = result.riskLevel.rawValue
                    entity.processingTime = result.processingTime
                    entity.createdAt = result.metadata.analysisDate
                    entity.sdkVersion = result.metadata.sdkVersion
                    entity.textLength = Int32(result.metadata.textLength)
                    
                    // Encode complex data as JSON
                    let encoder = JSONEncoder()
                    entity.keywordMatches = try encoder.encode(result.keywordMatches)
                    entity.exclusionMatches = try encoder.encode(result.exclusionMatches)
                    entity.recommendations = try encoder.encode(result.recommendations)
                    entity.metadata = try encoder.encode(result.metadata)
                    
                    try self.context.save()
                    
                    // Update cache
                    self.cacheQueue.async {
                        self.cache.setObject(result, forKey: result.id as NSString)
                    }
                    
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Get analysis result by ID
    public func getAnalysisResult(id: String) async throws -> ArbitrationAnalysisResult? {
        // Check cache first
        if let cachedResult = cache.object(forKey: id as NSString) {
            return cachedResult
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<AnalysisResultEntity> = AnalysisResultEntity.fetchRequest()
                    fetchRequest.predicate = NSPredicate(format: "id == %@", id)
                    
                    let entities = try self.context.fetch(fetchRequest)
                    
                    guard let entity = entities.first else {
                        continuation.resume(returning: nil)
                        return
                    }
                    
                    let result = try self.convertEntityToResult(entity)
                    
                    // Update cache
                    self.cacheQueue.async {
                        self.cache.setObject(result, forKey: id as NSString)
                    }
                    
                    continuation.resume(returning: result)
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Get all analysis results with optional filtering
    public func getAllAnalysisResults(
        limit: Int = 50,
        offset: Int = 0,
        sortBy: SortOption = .dateDescending,
        filter: ResultFilter? = nil
    ) async throws -> [ArbitrationAnalysisResult] {
        return try await withCheckedThrowingContinuation { continuation in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<AnalysisResultEntity> = AnalysisResultEntity.fetchRequest()
                    
                    // Apply filter
                    if let filter = filter {
                        fetchRequest.predicate = self.createPredicate(for: filter)
                    }
                    
                    // Apply sorting
                    fetchRequest.sortDescriptors = [self.createSortDescriptor(for: sortBy)]
                    
                    // Apply pagination
                    fetchRequest.fetchLimit = limit
                    fetchRequest.fetchOffset = offset
                    
                    let entities = try self.context.fetch(fetchRequest)
                    let results = try entities.map { try self.convertEntityToResult($0) }
                    
                    continuation.resume(returning: results)
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Delete analysis result
    public func deleteAnalysisResult(id: String) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<AnalysisResultEntity> = AnalysisResultEntity.fetchRequest()
                    fetchRequest.predicate = NSPredicate(format: "id == %@", id)
                    
                    let entities = try self.context.fetch(fetchRequest)
                    
                    for entity in entities {
                        self.context.delete(entity)
                    }
                    
                    try self.context.save()
                    
                    // Remove from cache
                    self.cache.removeObject(forKey: id as NSString)
                    
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Clear all analysis results
    public func clearAllAnalysisResults() async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<NSFetchRequestResult> = AnalysisResultEntity.fetchRequest()
                    let deleteRequest = NSBatchDeleteRequest(fetchRequest: fetchRequest)
                    
                    try self.context.execute(deleteRequest)
                    try self.context.save()
                    
                    // Clear cache
                    self.cache.removeAllObjects()
                    
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    // MARK: - User Preferences
    
    /// Save user preferences
    public func saveUserPreferences(_ preferences: UserPreferences) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    // Fetch or create preferences entity
                    let fetchRequest: NSFetchRequest<UserPreferencesEntity> = UserPreferencesEntity.fetchRequest()
                    let entities = try self.context.fetch(fetchRequest)
                    let entity = entities.first ?? UserPreferencesEntity(context: self.context)
                    
                    // Update entity
                    entity.language = preferences.language
                    entity.theme = preferences.theme.rawValue
                    entity.autoSave = preferences.autoSave
                    entity.biometricAuth = preferences.biometricAuth
                    
                    // Encode notifications settings
                    let encoder = JSONEncoder()
                    entity.notificationSettings = try encoder.encode(preferences.notifications)
                    
                    try self.context.save()
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Get user preferences
    public func getUserPreferences() async throws -> UserPreferences? {
        return try await withCheckedThrowingContinuation { continuation in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<UserPreferencesEntity> = UserPreferencesEntity.fetchRequest()
                    let entities = try self.context.fetch(fetchRequest)
                    
                    guard let entity = entities.first else {
                        continuation.resume(returning: nil)
                        return
                    }
                    
                    let decoder = JSONDecoder()
                    let notifications = try decoder.decode(NotificationSettings.self, from: entity.notificationSettings ?? Data())
                    
                    let preferences = UserPreferences(
                        language: entity.language ?? "en",
                        theme: AppTheme(rawValue: entity.theme ?? "system") ?? .system,
                        notifications: notifications,
                        autoSave: entity.autoSave,
                        biometricAuth: entity.biometricAuth
                    )
                    
                    continuation.resume(returning: preferences)
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    // MARK: - Analytics Events
    
    /// Save analytics event for offline sync
    public func saveAnalyticsEvent(_ event: AnalyticsEvent) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    let entity = AnalyticsEventEntity(context: self.context)
                    entity.id = UUID().uuidString
                    entity.eventType = event.eventType.rawValue
                    entity.timestamp = event.timestamp
                    entity.isSynced = false
                    
                    // Encode properties
                    let encoder = JSONEncoder()
                    entity.properties = try encoder.encode(event.properties)
                    
                    try self.context.save()
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Get unsynced analytics events
    public func getUnsyncedAnalyticsEvents() async throws -> [AnalyticsEvent] {
        return try await withCheckedThrowingContinuation { continuation in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<AnalyticsEventEntity> = AnalyticsEventEntity.fetchRequest()
                    fetchRequest.predicate = NSPredicate(format: "isSynced == %@", NSNumber(value: false))
                    fetchRequest.sortDescriptors = [NSSortDescriptor(keyPath: \AnalyticsEventEntity.timestamp, ascending: true)]
                    
                    let entities = try self.context.fetch(fetchRequest)
                    let events = try entities.map { entity -> AnalyticsEvent in
                        let decoder = JSONDecoder()
                        let properties = try decoder.decode([String: String].self, from: entity.properties ?? Data())
                        
                        let eventType = AnalyticsEventType(rawValue: entity.eventType ?? "") ?? .analysisStarted
                        
                        return AnalyticsEvent(eventType: eventType, properties: properties)
                    }
                    
                    continuation.resume(returning: events)
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    /// Mark analytics events as synced
    public func markAnalyticsEventsSynced(_ eventIds: [String]) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            context.perform {
                do {
                    let fetchRequest: NSFetchRequest<AnalyticsEventEntity> = AnalyticsEventEntity.fetchRequest()
                    fetchRequest.predicate = NSPredicate(format: "id IN %@", eventIds)
                    
                    let entities = try self.context.fetch(fetchRequest)
                    for entity in entities {
                        entity.isSynced = true
                    }
                    
                    try self.context.save()
                    continuation.resume(returning: ())
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    // MARK: - Statistics
    
    /// Get storage statistics
    public func getStorageStatistics() async throws -> StorageStatistics {
        return try await withCheckedThrowingContinuation { continuation in
            context.perform {
                do {
                    // Count analysis results
                    let analysisRequest: NSFetchRequest<AnalysisResultEntity> = AnalysisResultEntity.fetchRequest()
                    let totalAnalyses = try self.context.count(for: analysisRequest)
                    
                    // Count with arbitration
                    analysisRequest.predicate = NSPredicate(format: "hasArbitration == %@", NSNumber(value: true))
                    let analysesWithArbitration = try self.context.count(for: analysisRequest)
                    
                    // Count analytics events
                    let analyticsRequest: NSFetchRequest<AnalyticsEventEntity> = AnalyticsEventEntity.fetchRequest()
                    let totalEvents = try self.context.count(for: analyticsRequest)
                    
                    // Unsynced events
                    analyticsRequest.predicate = NSPredicate(format: "isSynced == %@", NSNumber(value: false))
                    let unsyncedEvents = try self.context.count(for: analyticsRequest)
                    
                    let statistics = StorageStatistics(
                        totalAnalyses: totalAnalyses,
                        analysesWithArbitration: analysesWithArbitration,
                        totalEvents: totalEvents,
                        unsyncedEvents: unsyncedEvents,
                        cacheHitRate: 0.0, // Calculate based on cache metrics
                        databaseSize: self.getDatabaseSize()
                    )
                    
                    continuation.resume(returning: statistics)
                } catch {
                    continuation.resume(throwing: ArbitrationError.storageError(error.localizedDescription))
                }
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func convertEntityToResult(_ entity: AnalysisResultEntity) throws -> ArbitrationAnalysisResult {
        let decoder = JSONDecoder()
        
        let keywordMatches = try decoder.decode([KeywordMatch].self, from: entity.keywordMatches ?? Data())
        let exclusionMatches = try decoder.decode([KeywordMatch].self, from: entity.exclusionMatches ?? Data())
        let recommendations = try decoder.decode([String].self, from: entity.recommendations ?? Data())
        let metadata = try decoder.decode(AnalysisMetadata.self, from: entity.metadata ?? Data())
        
        return ArbitrationAnalysisResult(
            id: entity.id ?? "",
            hasArbitration: entity.hasArbitration,
            confidence: entity.confidence,
            keywordMatches: keywordMatches,
            exclusionMatches: exclusionMatches,
            riskLevel: RiskLevel(rawValue: entity.riskLevel ?? "minimal") ?? .minimal,
            recommendations: recommendations,
            metadata: metadata
        )
    }
    
    private func createPredicate(for filter: ResultFilter) -> NSPredicate {
        var predicates: [NSPredicate] = []
        
        if let hasArbitration = filter.hasArbitration {
            predicates.append(NSPredicate(format: "hasArbitration == %@", NSNumber(value: hasArbitration)))
        }
        
        if let minConfidence = filter.minConfidence {
            predicates.append(NSPredicate(format: "confidence >= %f", minConfidence))
        }
        
        if let maxConfidence = filter.maxConfidence {
            predicates.append(NSPredicate(format: "confidence <= %f", maxConfidence))
        }
        
        if let riskLevel = filter.riskLevel {
            predicates.append(NSPredicate(format: "riskLevel == %@", riskLevel.rawValue))
        }
        
        if let dateFrom = filter.dateFrom {
            predicates.append(NSPredicate(format: "createdAt >= %@", dateFrom as NSDate))
        }
        
        if let dateTo = filter.dateTo {
            predicates.append(NSPredicate(format: "createdAt <= %@", dateTo as NSDate))
        }
        
        return NSCompoundPredicate(andPredicateWithSubpredicates: predicates)
    }
    
    private func createSortDescriptor(for sortOption: SortOption) -> NSSortDescriptor {
        switch sortOption {
        case .dateAscending:
            return NSSortDescriptor(keyPath: \AnalysisResultEntity.createdAt, ascending: true)
        case .dateDescending:
            return NSSortDescriptor(keyPath: \AnalysisResultEntity.createdAt, ascending: false)
        case .confidenceAscending:
            return NSSortDescriptor(keyPath: \AnalysisResultEntity.confidence, ascending: true)
        case .confidenceDescending:
            return NSSortDescriptor(keyPath: \AnalysisResultEntity.confidence, ascending: false)
        }
    }
    
    private func setupRemoteChangeNotifications() {
        NotificationCenter.default.publisher(for: .NSPersistentStoreRemoteChange)
            .sink { [weak self] _ in
                self?.handleRemoteChange()
            }
            .store(in: &cancellables)
    }
    
    private func handleRemoteChange() {
        // Handle remote changes for CloudKit sync if needed
        context.perform {
            try? self.context.save()
        }
    }
    
    private func getDatabaseSize() -> Int64 {
        // Calculate database file size
        let storeURL = container.persistentStoreDescriptions.first?.url
        if let url = storeURL,
           let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
           let size = attributes[.size] as? NSNumber {
            return size.int64Value
        }
        return 0
    }
}

// MARK: - Supporting Types

public enum SortOption: CaseIterable {
    case dateAscending
    case dateDescending
    case confidenceAscending
    case confidenceDescending
    
    public var displayName: String {
        switch self {
        case .dateAscending:
            return "Date (Oldest First)"
        case .dateDescending:
            return "Date (Newest First)"
        case .confidenceAscending:
            return "Confidence (Low to High)"
        case .confidenceDescending:
            return "Confidence (High to Low)"
        }
    }
}

public struct ResultFilter {
    public let hasArbitration: Bool?
    public let minConfidence: Double?
    public let maxConfidence: Double?
    public let riskLevel: RiskLevel?
    public let dateFrom: Date?
    public let dateTo: Date?
    
    public init(
        hasArbitration: Bool? = nil,
        minConfidence: Double? = nil,
        maxConfidence: Double? = nil,
        riskLevel: RiskLevel? = nil,
        dateFrom: Date? = nil,
        dateTo: Date? = nil
    ) {
        self.hasArbitration = hasArbitration
        self.minConfidence = minConfidence
        self.maxConfidence = maxConfidence
        self.riskLevel = riskLevel
        self.dateFrom = dateFrom
        self.dateTo = dateTo
    }
}

public struct StorageStatistics {
    public let totalAnalyses: Int
    public let analysesWithArbitration: Int
    public let totalEvents: Int
    public let unsyncedEvents: Int
    public let cacheHitRate: Double
    public let databaseSize: Int64
    
    public init(
        totalAnalyses: Int,
        analysesWithArbitration: Int,
        totalEvents: Int,
        unsyncedEvents: Int,
        cacheHitRate: Double,
        databaseSize: Int64
    ) {
        self.totalAnalyses = totalAnalyses
        self.analysesWithArbitration = analysesWithArbitration
        self.totalEvents = totalEvents
        self.unsyncedEvents = unsyncedEvents
        self.cacheHitRate = cacheHitRate
        self.databaseSize = databaseSize
    }
}