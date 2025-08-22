//
//  ARFrameworkModels.swift
//  AR Document Analysis Framework
//
//  Supporting models and utilities for the AR framework
//

import ARKit
import RealityKit
import SwiftUI
import Combine
import Foundation

// MARK: - Core AR Framework Models

/// Configuration for AR session setup
public struct ARSessionConfiguration {
    public var planeDetection: ARWorldTrackingConfiguration.PlaneDetection = [.horizontal, .vertical]
    public var environmentTexturing: ARWorldTrackingConfiguration.EnvironmentTexturing = .automatic
    public var isCollaborationEnabled: Bool = true
    public var userFaceTrackingEnabled: Bool = true
    public var isLightEstimationEnabled: Bool = true
    public var maximumNumberOfTrackedImages: Int = 10
    
    public init() {}
}

/// Document analysis result with AR positioning data
public struct ARDocumentAnalysis: Identifiable, Codable {
    public let id: UUID
    public let documentId: UUID
    public let cameraTransform: CodableTransform
    public let worldMappingStatus: ARFrame.WorldMappingStatus
    public let clauses: [ARClause]
    public let riskAssessment: ARRiskAssessment
    public let timestamp: Date
    public let confidence: Float
    
    public init(id: UUID = UUID(), documentId: UUID, cameraTransform: simd_float4x4, worldMappingStatus: ARFrame.WorldMappingStatus, clauses: [ARClause], riskAssessment: ARRiskAssessment, timestamp: Date = Date(), confidence: Float) {
        self.id = id
        self.documentId = documentId
        self.cameraTransform = CodableTransform(transform: cameraTransform)
        self.worldMappingStatus = worldMappingStatus
        self.clauses = clauses
        self.riskAssessment = riskAssessment
        self.timestamp = timestamp
        self.confidence = confidence
    }
}

/// AR-enhanced clause with spatial information
public struct ARClause: Identifiable, Codable {
    public let id: UUID
    public let content: String
    public let title: String
    public let section: String
    public let boundingBox: CodableRect
    public let worldPosition: CodableVector3
    public let riskLevel: RiskLevel
    public let confidence: Float
    public let keywords: [String]
    public let relationships: [ClauseRelationship]
    public let annotations: [ARAnnotation]
    public let visualElements: [ARVisualElement]
    
    public init(id: UUID = UUID(), content: String, title: String, section: String, boundingBox: CGRect, worldPosition: SIMD3<Float>, riskLevel: RiskLevel, confidence: Float, keywords: [String] = [], relationships: [ClauseRelationship] = [], annotations: [ARAnnotation] = [], visualElements: [ARVisualElement] = []) {
        self.id = id
        self.content = content
        self.title = title
        self.section = section
        self.boundingBox = CodableRect(rect: boundingBox)
        self.worldPosition = CodableVector3(vector: worldPosition)
        self.riskLevel = riskLevel
        self.confidence = confidence
        self.keywords = keywords
        self.relationships = relationships
        self.annotations = annotations
        self.visualElements = visualElements
    }
}

/// Risk assessment data for AR visualization
public struct ARRiskAssessment: Codable {
    public let overallRisk: RiskLevel
    public let riskDistribution: [RiskLevel: Int]
    public let criticalClauses: [UUID]
    public let riskFactors: [RiskFactor]
    public let mitigationSuggestions: [MitigationSuggestion]
    public let heatmapData: [RiskHeatmapPoint]
    
    public init(overallRisk: RiskLevel, riskDistribution: [RiskLevel: Int], criticalClauses: [UUID], riskFactors: [RiskFactor], mitigationSuggestions: [MitigationSuggestion], heatmapData: [RiskHeatmapPoint]) {
        self.overallRisk = overallRisk
        self.riskDistribution = riskDistribution
        self.criticalClauses = criticalClauses
        self.riskFactors = riskFactors
        self.mitigationSuggestions = mitigationSuggestions
        self.heatmapData = heatmapData
    }
}

/// Individual risk factor with spatial data
public struct RiskFactor: Identifiable, Codable {
    public let id: UUID
    public let type: RiskFactorType
    public let description: String
    public let severity: Float
    public let affectedClauses: [UUID]
    public let position: CodableVector3
    public let visualMarker: ARVisualElement
    
    public init(id: UUID = UUID(), type: RiskFactorType, description: String, severity: Float, affectedClauses: [UUID], position: SIMD3<Float>, visualMarker: ARVisualElement) {
        self.id = id
        self.type = type
        self.description = description
        self.severity = severity
        self.affectedClauses = affectedClauses
        self.position = CodableVector3(vector: position)
        self.visualMarker = visualMarker
    }
}

/// Types of risk factors that can be identified
public enum RiskFactorType: String, Codable, CaseIterable {
    case liability = "liability"
    case termination = "termination"
    case intellectual_property = "intellectual_property"
    case confidentiality = "confidentiality"
    case indemnification = "indemnification"
    case force_majeure = "force_majeure"
    case governing_law = "governing_law"
    case dispute_resolution = "dispute_resolution"
    case payment_terms = "payment_terms"
    case warranties = "warranties"
    
    public var displayName: String {
        switch self {
        case .liability: return "Liability"
        case .termination: return "Termination"
        case .intellectual_property: return "Intellectual Property"
        case .confidentiality: return "Confidentiality"
        case .indemnification: return "Indemnification"
        case .force_majeure: return "Force Majeure"
        case .governing_law: return "Governing Law"
        case .dispute_resolution: return "Dispute Resolution"
        case .payment_terms: return "Payment Terms"
        case .warranties: return "Warranties"
        }
    }
    
    public var color: UIColor {
        switch self {
        case .liability: return .systemRed
        case .termination: return .systemOrange
        case .intellectual_property: return .systemPurple
        case .confidentiality: return .systemBlue
        case .indemnification: return .systemPink
        case .force_majeure: return .systemGreen
        case .governing_law: return .systemTeal
        case .dispute_resolution: return .systemYellow
        case .payment_terms: return .systemIndigo
        case .warranties: return .systemGray
        }
    }
}

/// Mitigation suggestion with AR positioning
public struct MitigationSuggestion: Identifiable, Codable {
    public let id: UUID
    public let riskFactorId: UUID
    public let title: String
    public let description: String
    public let priority: Int
    public let estimatedImpact: Float
    public let suggestedChanges: [String]
    public let position: CodableVector3
    public let visualIndicator: ARVisualElement
    
    public init(id: UUID = UUID(), riskFactorId: UUID, title: String, description: String, priority: Int, estimatedImpact: Float, suggestedChanges: [String], position: SIMD3<Float>, visualIndicator: ARVisualElement) {
        self.id = id
        self.riskFactorId = riskFactorId
        self.title = title
        self.description = description
        self.priority = priority
        self.estimatedImpact = estimatedImpact
        self.suggestedChanges = suggestedChanges
        self.position = CodableVector3(vector: position)
        self.visualIndicator = visualIndicator
    }
}

/// Point data for risk heatmap visualization
public struct RiskHeatmapPoint: Codable {
    public let position: CodableVector3
    public let intensity: Float
    public let riskType: RiskFactorType
    public let radius: Float
    public let color: CodableColor
    
    public init(position: SIMD3<Float>, intensity: Float, riskType: RiskFactorType, radius: Float, color: UIColor) {
        self.position = CodableVector3(vector: position)
        self.intensity = intensity
        self.riskType = riskType
        self.radius = radius
        self.color = CodableColor(color: color)
    }
}

/// AR visual element for UI components
public struct ARVisualElement: Codable {
    public let type: VisualElementType
    public let position: CodableVector3
    public let scale: CodableVector3
    public let rotation: CodableQuaternion
    public let color: CodableColor
    public let opacity: Float
    public let isInteractive: Bool
    public let animationType: AnimationType?
    public let metadata: [String: String]
    
    public init(type: VisualElementType, position: SIMD3<Float>, scale: SIMD3<Float> = SIMD3<Float>(1, 1, 1), rotation: simd_quatf = simd_quatf(), color: UIColor, opacity: Float = 1.0, isInteractive: Bool = false, animationType: AnimationType? = nil, metadata: [String: String] = [:]) {
        self.type = type
        self.position = CodableVector3(vector: position)
        self.scale = CodableVector3(vector: scale)
        self.rotation = CodableQuaternion(quaternion: rotation)
        self.color = CodableColor(color: color)
        self.opacity = opacity
        self.isInteractive = isInteractive
        self.animationType = animationType
        self.metadata = metadata
    }
}

/// Types of visual elements in AR space
public enum VisualElementType: String, Codable, CaseIterable {
    case sphere = "sphere"
    case cube = "cube"
    case plane = "plane"
    case cylinder = "cylinder"
    case cone = "cone"
    case text = "text"
    case icon = "icon"
    case arrow = "arrow"
    case ring = "ring"
    case particle = "particle"
    case hologram = "hologram"
    case bubble = "bubble"
}

/// Animation types for AR elements
public enum AnimationType: String, Codable, CaseIterable {
    case pulse = "pulse"
    case rotate = "rotate"
    case float = "float"
    case scale = "scale"
    case fade = "fade"
    case bounce = "bounce"
    case shimmer = "shimmer"
    case glow = "glow"
    case none = "none"
}

/// Gesture interaction data
public struct ARGestureData {
    public let type: ARGestureType
    public let location: CGPoint
    public let worldPosition: SIMD3<Float>?
    public let velocity: CGPoint
    public let scale: Float
    public let rotation: Float
    public let timestamp: Date
    
    public init(type: ARGestureType, location: CGPoint, worldPosition: SIMD3<Float>? = nil, velocity: CGPoint = .zero, scale: Float = 1.0, rotation: Float = 0.0, timestamp: Date = Date()) {
        self.type = type
        self.location = location
        self.worldPosition = worldPosition
        self.velocity = velocity
        self.scale = scale
        self.rotation = rotation
        self.timestamp = timestamp
    }
}

/// Types of AR gestures
public enum ARGestureType: String, CaseIterable {
    case tap = "tap"
    case doubleTap = "doubleTap"
    case longPress = "longPress"
    case pinch = "pinch"
    case rotation = "rotation"
    case pan = "pan"
    case swipe = "swipe"
    case airTap = "airTap"
    case handPoint = "handPoint"
    case voiceCommand = "voiceCommand"
}

/// Voice command data structure
public struct VoiceCommand {
    public let command: String
    public let confidence: Float
    public let intent: CommandIntent
    public let entities: [String]
    public let timestamp: Date
    
    public init(command: String, confidence: Float, intent: CommandIntent, entities: [String] = [], timestamp: Date = Date()) {
        self.command = command
        self.confidence = confidence
        self.intent = intent
        self.entities = entities
        self.timestamp = timestamp
    }
}

/// Voice command intents
public enum CommandIntent: String, CaseIterable {
    case scan = "scan"
    case analyze = "analyze"
    case explain = "explain"
    case highlight = "highlight"
    case compare = "compare"
    case translate = "translate"
    case save = "save"
    case share = "share"
    case navigate = "navigate"
    case help = "help"
    case unknown = "unknown"
}

/// AR session state management
public class ARSessionState: ObservableObject {
    @Published public var isActive: Bool = false
    @Published public var trackingState: ARCamera.TrackingState = .notAvailable
    @Published public var worldMappingStatus: ARFrame.WorldMappingStatus = .notAvailable
    @Published public var anchorsCount: Int = 0
    @Published public var lastError: ARError?
    @Published public var frameRate: Float = 0.0
    @Published public var lightEstimate: ARLightEstimate?
    
    public init() {}
    
    public func update(with frame: ARFrame) {
        trackingState = frame.camera.trackingState
        worldMappingStatus = frame.worldMappingStatus
        lightEstimate = frame.lightEstimate
    }
}

/// Performance metrics for AR session
public struct ARPerformanceMetrics {
    public let frameRate: Float
    public let cpuUsage: Float
    public let memoryUsage: Float
    public let thermalState: ProcessInfo.ThermalState
    public let batteryLevel: Float
    public let renderingTime: TimeInterval
    public let trackingQuality: Float
    
    public init(frameRate: Float, cpuUsage: Float, memoryUsage: Float, thermalState: ProcessInfo.ThermalState, batteryLevel: Float, renderingTime: TimeInterval, trackingQuality: Float) {
        self.frameRate = frameRate
        self.cpuUsage = cpuUsage
        self.memoryUsage = memoryUsage
        self.thermalState = thermalState
        self.batteryLevel = batteryLevel
        self.renderingTime = renderingTime
        self.trackingQuality = trackingQuality
    }
}

// MARK: - Collaboration Models

/// Peer information for AR collaboration
public struct ARCollaborationPeer: Identifiable, Codable {
    public let id: UUID
    public let deviceId: String
    public let displayName: String
    public let avatar: ARAvatar?
    public let capabilities: [ARCapability]
    public let position: CodableVector3?
    public let orientation: CodableQuaternion?
    public let isActive: Bool
    
    public init(id: UUID = UUID(), deviceId: String, displayName: String, avatar: ARAvatar? = nil, capabilities: [ARCapability] = [], position: SIMD3<Float>? = nil, orientation: simd_quatf? = nil, isActive: Bool = true) {
        self.id = id
        self.deviceId = deviceId
        self.displayName = displayName
        self.avatar = avatar
        self.capabilities = capabilities
        self.position = position.map(CodableVector3.init)
        self.orientation = orientation.map(CodableQuaternion.init)
        self.isActive = isActive
    }
}

/// AR capabilities for collaboration
public enum ARCapability: String, Codable, CaseIterable {
    case worldTracking = "worldTracking"
    case faceTracking = "faceTracking"
    case handTracking = "handTracking"
    case bodyTracking = "bodyTracking"
    case lightEstimation = "lightEstimation"
    case peopleOcclusion = "peopleOcclusion"
    case collaboration = "collaboration"
    case screenSharing = "screenSharing"
    case voiceChat = "voiceChat"
    case spatialAudio = "spatialAudio"
}

/// Avatar representation in AR space
public struct ARAvatar: Codable {
    public let style: AvatarStyle
    public let color: CodableColor
    public let accessories: [AvatarAccessory]
    public let animations: [AvatarAnimation]
    
    public init(style: AvatarStyle, color: UIColor, accessories: [AvatarAccessory] = [], animations: [AvatarAnimation] = []) {
        self.style = style
        self.color = CodableColor(color: color)
        self.accessories = accessories
        self.animations = animations
    }
}

/// Avatar style options
public enum AvatarStyle: String, Codable, CaseIterable {
    case hologram = "hologram"
    case robot = "robot"
    case abstract = "abstract"
    case geometric = "geometric"
    case particle = "particle"
}

/// Avatar accessories
public struct AvatarAccessory: Codable {
    public let type: AccessoryType
    public let position: CodableVector3
    public let color: CodableColor
    
    public init(type: AccessoryType, position: SIMD3<Float>, color: UIColor) {
        self.type = type
        self.position = CodableVector3(vector: position)
        self.color = CodableColor(color: color)
    }
}

/// Types of avatar accessories
public enum AccessoryType: String, Codable, CaseIterable {
    case hat = "hat"
    case glasses = "glasses"
    case badge = "badge"
    case aura = "aura"
    case trail = "trail"
}

/// Avatar animations
public struct AvatarAnimation: Codable {
    public let type: AnimationType
    public let duration: TimeInterval
    public let repeatCount: Int
    public let parameters: [String: Float]
    
    public init(type: AnimationType, duration: TimeInterval, repeatCount: Int = 1, parameters: [String: Float] = [:]) {
        self.type = type
        self.duration = duration
        self.repeatCount = repeatCount
        self.parameters = parameters
    }
}

/// Shared workspace data
public struct ARWorkspace: Identifiable, Codable {
    public let id: UUID
    public let name: String
    public let documents: [UUID]
    public let participants: [UUID]
    public let sharedElements: [ARSharedElement]
    public let permissions: WorkspacePermissions
    public let createdAt: Date
    public let lastModified: Date
    
    public init(id: UUID = UUID(), name: String, documents: [UUID] = [], participants: [UUID] = [], sharedElements: [ARSharedElement] = [], permissions: WorkspacePermissions = WorkspacePermissions(), createdAt: Date = Date(), lastModified: Date = Date()) {
        self.id = id
        self.name = name
        self.documents = documents
        self.participants = participants
        self.sharedElements = sharedElements
        self.permissions = permissions
        self.createdAt = createdAt
        self.lastModified = lastModified
    }
}

/// Shared element in AR workspace
public struct ARSharedElement: Identifiable, Codable {
    public let id: UUID
    public let type: SharedElementType
    public let position: CodableVector3
    public let data: Data
    public let ownerId: UUID
    public let permissions: ElementPermissions
    public let timestamp: Date
    
    public init(id: UUID = UUID(), type: SharedElementType, position: SIMD3<Float>, data: Data, ownerId: UUID, permissions: ElementPermissions = ElementPermissions(), timestamp: Date = Date()) {
        self.id = id
        self.type = type
        self.position = CodableVector3(vector: position)
        self.data = data
        self.ownerId = ownerId
        self.permissions = permissions
        self.timestamp = timestamp
    }
}

/// Types of shared elements
public enum SharedElementType: String, Codable, CaseIterable {
    case annotation = "annotation"
    case highlight = "highlight"
    case comment = "comment"
    case pointer = "pointer"
    case document = "document"
    case media = "media"
    case drawing = "drawing"
    case bookmark = "bookmark"
}

/// Workspace permissions
public struct WorkspacePermissions: Codable {
    public let canEdit: Bool
    public let canAddElements: Bool
    public let canRemoveElements: Bool
    public let canInviteUsers: Bool
    public let canModifyPermissions: Bool
    
    public init(canEdit: Bool = true, canAddElements: Bool = true, canRemoveElements: Bool = true, canInviteUsers: Bool = false, canModifyPermissions: Bool = false) {
        self.canEdit = canEdit
        self.canAddElements = canAddElements
        self.canRemoveElements = canRemoveElements
        self.canInviteUsers = canInviteUsers
        self.canModifyPermissions = canModifyPermissions
    }
}

/// Element permissions
public struct ElementPermissions: Codable {
    public let canView: Bool
    public let canEdit: Bool
    public let canDelete: Bool
    public let canMove: Bool
    
    public init(canView: Bool = true, canEdit: Bool = true, canDelete: Bool = true, canMove: Bool = true) {
        self.canView = canView
        self.canEdit = canEdit
        self.canDelete = canDelete
        self.canMove = canMove
    }
}

// MARK: - Analytics and Insights Models

/// Usage analytics for AR features
public struct ARAnalytics {
    public let sessionDuration: TimeInterval
    public let gesturesUsed: [ARGestureType: Int]
    public let voiceCommandsUsed: [CommandIntent: Int]
    public let documentsScanned: Int
    public let clausesAnalyzed: Int
    public let risksIdentified: Int
    public let collaborationEvents: Int
    public let performanceMetrics: ARPerformanceMetrics
    
    public init(sessionDuration: TimeInterval, gesturesUsed: [ARGestureType: Int], voiceCommandsUsed: [CommandIntent: Int], documentsScanned: Int, clausesAnalyzed: Int, risksIdentified: Int, collaborationEvents: Int, performanceMetrics: ARPerformanceMetrics) {
        self.sessionDuration = sessionDuration
        self.gesturesUsed = gesturesUsed
        self.voiceCommandsUsed = voiceCommandsUsed
        self.documentsScanned = documentsScanned
        self.clausesAnalyzed = clausesAnalyzed
        self.risksIdentified = risksIdentified
        self.collaborationEvents = collaborationEvents
        self.performanceMetrics = performanceMetrics
    }
}

/// User insights from AR interactions
public struct ARUserInsights {
    public let preferredInteractionMethods: [ARGestureType]
    public let mostUsedFeatures: [String]
    public let averageSessionDuration: TimeInterval
    public let learningProgress: Float
    public let expertiseLevel: ExpertiseLevel
    public let recommendations: [String]
    
    public init(preferredInteractionMethods: [ARGestureType], mostUsedFeatures: [String], averageSessionDuration: TimeInterval, learningProgress: Float, expertiseLevel: ExpertiseLevel, recommendations: [String]) {
        self.preferredInteractionMethods = preferredInteractionMethods
        self.mostUsedFeatures = mostUsedFeatures
        self.averageSessionDuration = averageSessionDuration
        self.learningProgress = learningProgress
        self.expertiseLevel = expertiseLevel
        self.recommendations = recommendations
    }
}

/// User expertise levels
public enum ExpertiseLevel: String, Codable, CaseIterable {
    case beginner = "beginner"
    case intermediate = "intermediate"
    case advanced = "advanced"
    case expert = "expert"
    
    public var displayName: String {
        switch self {
        case .beginner: return "Beginner"
        case .intermediate: return "Intermediate"
        case .advanced: return "Advanced"
        case .expert: return "Expert"
        }
    }
}

// MARK: - Codable Helper Types

/// Codable wrapper for simd_float4x4
public struct CodableTransform: Codable {
    public let matrix: [[Float]]
    
    public init(transform: simd_float4x4) {
        self.matrix = [
            [transform.columns.0.x, transform.columns.0.y, transform.columns.0.z, transform.columns.0.w],
            [transform.columns.1.x, transform.columns.1.y, transform.columns.1.z, transform.columns.1.w],
            [transform.columns.2.x, transform.columns.2.y, transform.columns.2.z, transform.columns.2.w],
            [transform.columns.3.x, transform.columns.3.y, transform.columns.3.z, transform.columns.3.w]
        ]
    }
    
    public var simdTransform: simd_float4x4 {
        return simd_float4x4(
            SIMD4<Float>(matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]),
            SIMD4<Float>(matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]),
            SIMD4<Float>(matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]),
            SIMD4<Float>(matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3])
        )
    }
}

/// Codable wrapper for SIMD3<Float>
public struct CodableVector3: Codable {
    public let x: Float
    public let y: Float
    public let z: Float
    
    public init(vector: SIMD3<Float>) {
        self.x = vector.x
        self.y = vector.y
        self.z = vector.z
    }
    
    public var simdVector: SIMD3<Float> {
        return SIMD3<Float>(x, y, z)
    }
}

/// Codable wrapper for simd_quatf
public struct CodableQuaternion: Codable {
    public let x: Float
    public let y: Float
    public let z: Float
    public let w: Float
    
    public init(quaternion: simd_quatf = simd_quatf()) {
        self.x = quaternion.vector.x
        self.y = quaternion.vector.y
        self.z = quaternion.vector.z
        self.w = quaternion.vector.w
    }
    
    public var simdQuaternion: simd_quatf {
        return simd_quatf(ix: x, iy: y, iz: z, r: w)
    }
}

/// Codable wrapper for CGRect
public struct CodableRect: Codable {
    public let x: Double
    public let y: Double
    public let width: Double
    public let height: Double
    
    public init(rect: CGRect) {
        self.x = rect.origin.x
        self.y = rect.origin.y
        self.width = rect.size.width
        self.height = rect.size.height
    }
    
    public var cgRect: CGRect {
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

/// Codable wrapper for UIColor
public struct CodableColor: Codable {
    public let red: Double
    public let green: Double
    public let blue: Double
    public let alpha: Double
    
    public init(color: UIColor) {
        var r: CGFloat = 0
        var g: CGFloat = 0
        var b: CGFloat = 0
        var a: CGFloat = 0
        
        color.getRed(&r, green: &g, blue: &b, alpha: &a)
        
        self.red = Double(r)
        self.green = Double(g)
        self.blue = Double(b)
        self.alpha = Double(a)
    }
    
    public var uiColor: UIColor {
        return UIColor(red: red, green: green, blue: blue, alpha: alpha)
    }
}

// MARK: - Extensions for ARFrame.WorldMappingStatus

extension ARFrame.WorldMappingStatus: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .notAvailable:
            try container.encode("notAvailable")
        case .limited:
            try container.encode("limited")
        case .extending:
            try container.encode("extending")
        case .mapped:
            try container.encode("mapped")
        @unknown default:
            try container.encode("unknown")
        }
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(String.self)
        
        switch value {
        case "notAvailable":
            self = .notAvailable
        case "limited":
            self = .limited
        case "extending":
            self = .extending
        case "mapped":
            self = .mapped
        default:
            self = .notAvailable
        }
    }
}

// MARK: - Utility Extensions

extension RiskLevel: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(self.rawValue)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let value = try container.decode(Int.self)
        self = RiskLevel(rawValue: value) ?? .low
    }
}

// MARK: - Validation and Error Handling

/// Errors that can occur in the AR framework
public enum ARFrameworkError: Error, LocalizedError {
    case sessionNotInitialized
    case unsupportedDevice
    case cameraAccessDenied
    case worldTrackingFailed
    case collaborationSetupFailed
    case documentProcessingFailed
    case invalidConfiguration
    case networkError(String)
    case dataCorruption
    case insufficientMemory
    case thermalThrottling
    
    public var errorDescription: String? {
        switch self {
        case .sessionNotInitialized:
            return "AR session not initialized"
        case .unsupportedDevice:
            return "This device does not support AR features"
        case .cameraAccessDenied:
            return "Camera access is required for AR functionality"
        case .worldTrackingFailed:
            return "World tracking failed - please check lighting conditions"
        case .collaborationSetupFailed:
            return "Failed to set up collaboration session"
        case .documentProcessingFailed:
            return "Failed to process document"
        case .invalidConfiguration:
            return "Invalid AR configuration"
        case .networkError(let message):
            return "Network error: \(message)"
        case .dataCorruption:
            return "Data corruption detected"
        case .insufficientMemory:
            return "Insufficient memory for AR operations"
        case .thermalThrottling:
            return "Device is overheating - AR features may be limited"
        }
    }
}

/// Validation utilities
public struct ARFrameworkValidator {
    
    /// Validate AR session configuration
    public static func validate(configuration: ARSessionConfiguration) -> Result<Void, ARFrameworkError> {
        // Check device capabilities
        if !ARWorldTrackingConfiguration.isSupported {
            return .failure(.unsupportedDevice)
        }
        
        // Validate collaboration settings
        if configuration.isCollaborationEnabled && !ARWorldTrackingConfiguration.supportsCollaboration {
            return .failure(.invalidConfiguration)
        }
        
        // Validate face tracking settings
        if configuration.userFaceTrackingEnabled && !ARWorldTrackingConfiguration.supportsUserFaceTracking {
            return .failure(.invalidConfiguration)
        }
        
        return .success(())
    }
    
    /// Validate document analysis data
    public static func validate(analysis: ARDocumentAnalysis) -> Bool {
        return !analysis.clauses.isEmpty &&
               analysis.confidence > 0.0 &&
               analysis.confidence <= 1.0
    }
    
    /// Validate AR workspace data
    public static func validate(workspace: ARWorkspace) -> Bool {
        return !workspace.name.isEmpty &&
               workspace.participants.count > 0
    }
}

// MARK: - Performance Monitoring

/// Performance monitor for AR operations
public class ARPerformanceMonitor: ObservableObject {
    @Published public var currentMetrics = ARPerformanceMetrics(
        frameRate: 0,
        cpuUsage: 0,
        memoryUsage: 0,
        thermalState: .nominal,
        batteryLevel: 1.0,
        renderingTime: 0,
        trackingQuality: 1.0
    )
    
    private var timer: Timer?
    
    public init() {}
    
    public func startMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateMetrics()
        }
    }
    
    public func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updateMetrics() {
        let processInfo = ProcessInfo.processInfo
        
        // Get battery level
        UIDevice.current.isBatteryMonitoringEnabled = true
        let batteryLevel = UIDevice.current.batteryLevel
        
        // Update metrics (simplified - full implementation would use more detailed monitoring)
        let updatedMetrics = ARPerformanceMetrics(
            frameRate: 60.0, // Would measure actual frame rate
            cpuUsage: Float(processInfo.processorCount) * 0.1, // Simplified CPU usage
            memoryUsage: 0.5, // Would measure actual memory usage
            thermalState: processInfo.thermalState,
            batteryLevel: batteryLevel >= 0 ? batteryLevel : 1.0,
            renderingTime: 0.016, // 60fps = ~16ms per frame
            trackingQuality: 1.0 // Would measure actual tracking quality
        )
        
        DispatchQueue.main.async {
            self.currentMetrics = updatedMetrics
        }
    }
}

// MARK: - Debug and Logging

/// Debug utilities for AR framework
public struct ARDebugUtils {
    
    /// Log AR session events
    public static func logEvent(_ event: String, data: [String: Any] = [:]) {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        print("[\(timestamp)] AR Event: \(event)")
        
        if !data.isEmpty {
            for (key, value) in data {
                print("  \(key): \(value)")
            }
        }
    }
    
    /// Validate AR anchor positions
    public static func validateAnchorPositions(_ anchors: [ARAnchor]) -> [String] {
        var issues: [String] = []
        
        for anchor in anchors {
            let position = anchor.transform.columns.3
            
            // Check for extreme positions
            if abs(position.x) > 100 || abs(position.y) > 100 || abs(position.z) > 100 {
                issues.append("Anchor \(anchor.identifier) has extreme position: \(position)")
            }
            
            // Check for NaN values
            if position.x.isNaN || position.y.isNaN || position.z.isNaN {
                issues.append("Anchor \(anchor.identifier) has NaN position values")
            }
        }
        
        return issues
    }
    
    /// Generate AR session report
    public static func generateSessionReport(
        sessionDuration: TimeInterval,
        anchorsCreated: Int,
        gesturesProcessed: Int,
        errors: [Error]
    ) -> String {
        var report = "AR Session Report\n"
        report += "=================\n"
        report += "Duration: \(String(format: "%.2f", sessionDuration))s\n"
        report += "Anchors Created: \(anchorsCreated)\n"
        report += "Gestures Processed: \(gesturesProcessed)\n"
        report += "Errors: \(errors.count)\n"
        
        if !errors.isEmpty {
            report += "\nError Details:\n"
            for (index, error) in errors.enumerated() {
                report += "\(index + 1). \(error.localizedDescription)\n"
            }
        }
        
        return report
    }
}