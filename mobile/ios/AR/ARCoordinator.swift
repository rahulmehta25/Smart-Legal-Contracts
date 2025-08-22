//
//  ARCoordinator.swift
//  AR Document Analysis Framework
//
//  Main coordinator that integrates all AR components
//

import ARKit
import RealityKit
import SwiftUI
import Combine

@available(iOS 13.0, *)
public class ARCoordinator: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isARSessionActive = false
    @Published public var currentDocument: AnalyzedDocument?
    @Published public var sessionState = ARSessionState()
    @Published public var activeFeatures: Set<ARFeature> = []
    
    // AR Components
    private var documentScanner: DocumentScanner?
    private var clauseVisualizer: ClauseVisualizer?
    private var riskHeatmap: RiskHeatmap?
    private var collaborationManager: ARCollaborationManager?
    private var aiAssistant: AIAssistant?
    private var gestureController: ARGestureController?
    
    // AR View
    private var arView: ARView?
    private var performanceMonitor = ARPerformanceMonitor()
    
    // Configuration
    private var configuration = ARSessionConfiguration()
    
    // Analytics
    private var analytics = ARGestureAnalytics()
    
    // Subscriptions
    private var cancellables = Set<AnyCancellable>()
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupComponents()
        setupBindings()
    }
    
    deinit {
        stopARSession()
    }
    
    // MARK: - Public Methods
    
    public func startARSession(in arView: ARView, with configuration: ARSessionConfiguration = ARSessionConfiguration()) {
        self.arView = arView
        self.configuration = configuration
        
        guard validateConfiguration() else {
            print("Invalid AR configuration")
            return
        }
        
        setupARView()
        initializeComponents()
        startSession()
        
        isARSessionActive = true
        performanceMonitor.startMonitoring()
        analytics.startSession()
        
        ARDebugUtils.logEvent("AR session started")
    }
    
    public func stopARSession() {
        deactivateAllFeatures()
        
        arView?.session.pause()
        performanceMonitor.stopMonitoring()
        
        isARSessionActive = false
        ARDebugUtils.logEvent("AR session stopped")
    }
    
    public func activateFeature(_ feature: ARFeature) {
        guard isARSessionActive, let arView = arView else { return }
        
        switch feature {
        case .documentScanning:
            activateDocumentScanning()
        case .clauseVisualization:
            activateClauseVisualization()
        case .riskHeatmap:
            activateRiskHeatmap()
        case .collaboration:
            activateCollaboration()
        case .aiAssistant:
            activateAIAssistant()
        case .gestureControl:
            activateGestureControl()
        }
        
        activeFeatures.insert(feature)
        ARDebugUtils.logEvent("Feature activated", data: ["feature": feature.rawValue])
    }
    
    public func deactivateFeature(_ feature: ARFeature) {
        guard activeFeatures.contains(feature) else { return }
        
        switch feature {
        case .documentScanning:
            documentScanner?.stopScanning()
        case .clauseVisualization:
            clauseVisualizer?.stopVisualization()
        case .riskHeatmap:
            riskHeatmap?.deactivateHeatmap()
        case .collaboration:
            collaborationManager?.stopSession()
        case .aiAssistant:
            aiAssistant?.deactivateAssistant()
        case .gestureControl:
            gestureController?.deactivateAllControls()
        }
        
        activeFeatures.remove(feature)
        ARDebugUtils.logEvent("Feature deactivated", data: ["feature": feature.rawValue])
    }
    
    public func scanDocument(at location: CGPoint) async -> ScannedDocument? {
        guard let scanner = documentScanner else { return nil }
        return await scanner.captureDocument(at: location)
    }
    
    public func analyzeDocument(_ scannedDocument: ScannedDocument) -> AnalyzedDocument? {
        // Convert scanned document to analyzed document
        let analyzedDoc = convertToAnalyzedDocument(scannedDocument)
        currentDocument = analyzedDoc
        
        // Update visualizations
        if activeFeatures.contains(.clauseVisualization) {
            updateClauseVisualization(analyzedDoc)
        }
        
        if activeFeatures.contains(.riskHeatmap) {
            updateRiskHeatmap(analyzedDoc)
        }
        
        return analyzedDoc
    }
    
    public func shareDocument(_ document: AnalyzedDocument) {
        collaborationManager?.shareDocument(document)
    }
    
    public func getSessionAnalytics() -> GestureAnalyticsReport? {
        return gestureController?.getSessionAnalytics()
    }
    
    // MARK: - Private Methods
    
    private func setupComponents() {
        documentScanner = DocumentScanner()
        clauseVisualizer = ClauseVisualizer()
        riskHeatmap = RiskHeatmap()
        collaborationManager = ARCollaborationManager()
        aiAssistant = AIAssistant()
        gestureController = ARGestureController()
    }
    
    private func setupBindings() {
        // Monitor performance
        performanceMonitor.$currentMetrics
            .sink { [weak self] metrics in
                self?.handlePerformanceUpdate(metrics)
            }
            .store(in: &cancellables)
        
        // Monitor document scanner
        documentScanner?.$scannedDocuments
            .sink { [weak self] documents in
                if let latest = documents.last {
                    _ = self?.analyzeDocument(latest)
                }
            }
            .store(in: &cancellables)
        
        // Monitor gesture controller
        gestureController?.$currentGesture
            .sink { [weak self] gesture in
                if let gesture = gesture {
                    self?.handleGesture(gesture)
                }
            }
            .store(in: &cancellables)
    }
    
    private func validateConfiguration() -> Bool {
        let validationResult = ARFrameworkValidator.validate(configuration: configuration)
        
        switch validationResult {
        case .success:
            return true
        case .failure(let error):
            print("Configuration validation failed: \(error)")
            return false
        }
    }
    
    private func setupARView() {
        guard let arView = arView else { return }
        
        let arConfiguration = ARWorldTrackingConfiguration()
        arConfiguration.planeDetection = configuration.planeDetection
        arConfiguration.environmentTexturing = configuration.environmentTexturing
        arConfiguration.isCollaborationEnabled = configuration.isCollaborationEnabled
        
        if configuration.userFaceTrackingEnabled && ARWorldTrackingConfiguration.supportsUserFaceTracking {
            arConfiguration.userFaceTrackingEnabled = true
        }
        
        arView.session.run(arConfiguration, options: [.removeExistingAnchors, .resetTracking])
        arView.session.delegate = self
    }
    
    private func initializeComponents() {
        // Set up component delegates
        gestureController?.gestureDelegate = self
        gestureController?.voiceDelegate = self
        gestureController?.handTrackingDelegate = self
    }
    
    private func startSession() {
        // Start monitoring session state
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            guard let self = self, self.isARSessionActive else {
                timer.invalidate()
                return
            }
            
            if let frame = self.arView?.session.currentFrame {
                self.sessionState.update(with: frame)
            }
        }
    }
    
    private func activateDocumentScanning() {
        guard let arView = arView, let scanner = documentScanner else { return }
        scanner.startScanning(in: arView)
    }
    
    private func activateClauseVisualization() {
        guard let arView = arView,
              let visualizer = clauseVisualizer,
              let document = currentDocument else { return }
        
        visualizer.startVisualization(for: document, in: arView)
    }
    
    private func activateRiskHeatmap() {
        guard let arView = arView,
              let heatmap = riskHeatmap,
              let document = currentDocument else { return }
        
        heatmap.activateHeatmap(for: document, in: arView)
    }
    
    private func activateCollaboration() {
        guard let arView = arView,
              let collaboration = collaborationManager else { return }
        
        collaboration.startCollaborationSession(in: arView)
    }
    
    private func activateAIAssistant() {
        guard let arView = arView,
              let assistant = aiAssistant else { return }
        
        assistant.activateAssistant(in: arView, with: currentDocument)
    }
    
    private func activateGestureControl() {
        guard let arView = arView,
              let controller = gestureController else { return }
        
        controller.activateGestureRecognition(in: arView)
        controller.activateVoiceControl()
        controller.activateHandTracking()
    }
    
    private func deactivateAllFeatures() {
        for feature in activeFeatures {
            deactivateFeature(feature)
        }
    }
    
    private func updateClauseVisualization(_ document: AnalyzedDocument) {
        guard let arView = arView, let visualizer = clauseVisualizer else { return }
        
        visualizer.stopVisualization()
        visualizer.startVisualization(for: document, in: arView)
    }
    
    private func updateRiskHeatmap(_ document: AnalyzedDocument) {
        guard let arView = arView, let heatmap = riskHeatmap else { return }
        
        let riskData = document.clauses.map { clause in
            RiskDataPoint(
                position: calculateClausePosition(clause),
                riskLevel: clause.riskLevel,
                confidence: clause.confidence,
                intensity: Float(clause.riskLevel.rawValue) / 4.0,
                metadata: RiskMetadata(
                    clauseTitle: clause.title,
                    section: clause.section,
                    content: clause.content
                )
            )
        }
        
        heatmap.updateRiskData(riskData)
    }
    
    private func convertToAnalyzedDocument(_ scannedDocument: ScannedDocument) -> AnalyzedDocument {
        // Extract clauses from OCR result
        let clauses = extractClausesFromOCR(scannedDocument.ocrResult)
        
        return AnalyzedDocument(
            title: "Scanned Document",
            clauses: clauses
        )
    }
    
    private func extractClausesFromOCR(_ ocrResult: OCRResult) -> [ClauseData] {
        var clauses: [ClauseData] = []
        
        // Simple clause extraction based on text blocks
        for (index, block) in ocrResult.blocks.enumerated() {
            let clause = ClauseData(
                title: "Clause \(index + 1)",
                content: block.text,
                section: "Section \((index / 3) + 1)",
                order: index,
                riskLevel: determineRiskLevel(for: block.text),
                confidence: block.confidence,
                relationships: []
            )
            clauses.append(clause)
        }
        
        return clauses
    }
    
    private func determineRiskLevel(for text: String) -> RiskLevel {
        let lowercaseText = text.lowercased()
        
        // Simple risk assessment based on keywords
        if lowercaseText.contains("unlimited liability") ||
           lowercaseText.contains("indemnify") ||
           lowercaseText.contains("hold harmless") {
            return .critical
        } else if lowercaseText.contains("liability") ||
                  lowercaseText.contains("damages") ||
                  lowercaseText.contains("terminate") {
            return .high
        } else if lowercaseText.contains("warranty") ||
                  lowercaseText.contains("guarantee") {
            return .medium
        } else {
            return .low
        }
    }
    
    private func calculateClausePosition(_ clause: ClauseData) -> SIMD3<Float> {
        let sectionHash = Float(clause.section.hashValue % 360) * .pi / 180
        let orderOffset = Float(clause.order) * 0.1
        
        let x = cos(sectionHash) * (0.4 + orderOffset * 0.05)
        let y = Float(clause.riskLevel.rawValue - 1) * 0.15
        let z = sin(sectionHash) * (0.4 + orderOffset * 0.05)
        
        return SIMD3<Float>(x, y, z)
    }
    
    private func handlePerformanceUpdate(_ metrics: ARPerformanceMetrics) {
        // Handle performance issues
        if metrics.thermalState != .nominal {
            print("Warning: Device thermal state is \(metrics.thermalState)")
            
            if metrics.thermalState == .critical {
                // Reduce AR features to prevent overheating
                if activeFeatures.contains(.riskHeatmap) {
                    deactivateFeature(.riskHeatmap)
                }
            }
        }
        
        if metrics.frameRate < 30 {
            print("Warning: Low frame rate: \(metrics.frameRate)")
        }
        
        if metrics.memoryUsage > 0.8 {
            print("Warning: High memory usage: \(metrics.memoryUsage)")
        }
    }
    
    private func handleGesture(_ gestureType: ARGestureType) {
        analytics.trackGesture(gestureType)
        
        // Handle common gestures
        switch gestureType {
        case .doubleTap:
            if !activeFeatures.contains(.aiAssistant) {
                activateFeature(.aiAssistant)
            }
        case .longPress:
            if activeFeatures.contains(.clauseVisualization) {
                clauseVisualizer?.animateDocumentFlow()
            }
        default:
            break
        }
    }
}

// MARK: - ARSessionDelegate

extension ARCoordinator: ARSessionDelegate {
    public func session(_ session: ARSession, didFailWithError error: Error) {
        print("AR Session failed: \(error)")
        
        if let arError = error as? ARError {
            switch arError.errorCode {
            case 102: // Unsupported configuration
                print("AR configuration not supported")
            case 200: // World tracking failed
                print("World tracking failed")
            default:
                print("AR Error: \(arError.localizedDescription)")
            }
        }
    }
    
    public func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        sessionState.trackingState = camera.trackingState
        
        switch camera.trackingState {
        case .normal:
            print("AR tracking normal")
        case .limited(let reason):
            print("AR tracking limited: \(reason)")
        case .notAvailable:
            print("AR tracking not available")
        }
    }
    
    public func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        sessionState.anchorsCount += anchors.count
        
        // Validate anchor positions
        let issues = ARDebugUtils.validateAnchorPositions(anchors)
        if !issues.isEmpty {
            print("Anchor validation issues: \(issues)")
        }
    }
    
    public func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        sessionState.anchorsCount -= anchors.count
    }
}

// MARK: - Gesture Delegates

extension ARCoordinator: ARGestureDelegate {
    public func didReceiveGesture(_ gesture: ARGestureData) {
        analytics.trackGesture(gesture.type)
        
        // Route gesture to appropriate component
        if activeFeatures.contains(.clauseVisualization) {
            handleClauseVisualizationGesture(gesture)
        }
        
        if activeFeatures.contains(.riskHeatmap) {
            handleRiskHeatmapGesture(gesture)
        }
        
        if activeFeatures.contains(.aiAssistant) {
            handleAIAssistantGesture(gesture)
        }
    }
    
    private func handleClauseVisualizationGesture(_ gesture: ARGestureData) {
        guard let visualizer = clauseVisualizer else { return }
        
        switch gesture.type {
        case .tap:
            if let worldPos = gesture.worldPosition {
                // Find nearest clause and highlight it
                if let document = currentDocument {
                    let nearestClause = findNearestClause(to: worldPos, in: document)
                    if let clause = nearestClause {
                        visualizer.highlightClause(clause.id)
                    }
                }
            }
        case .pinch:
            // Switch visualization mode on pinch
            let currentMode = visualizer.visualizationMode
            let nextMode: VisualizationMode
            
            switch currentMode {
            case .hierarchical:
                nextMode = .network
            case .network:
                nextMode = .timeline
            case .timeline:
                nextMode = .riskBased
            case .riskBased:
                nextMode = .comparative
            case .comparative:
                nextMode = .hierarchical
            }
            
            visualizer.switchVisualizationMode(nextMode)
        default:
            break
        }
    }
    
    private func handleRiskHeatmapGesture(_ gesture: ARGestureData) {
        guard let heatmap = riskHeatmap else { return }
        
        switch gesture.type {
        case .pan:
            // Adjust heatmap intensity based on pan velocity
            let velocity = sqrt(pow(gesture.velocity.x, 2) + pow(gesture.velocity.y, 2))
            let intensityAdjustment = Float(velocity / 1000.0)
            let newIntensity = heatmap.heatmapIntensity + intensityAdjustment * 0.1
            heatmap.updateIntensity(max(0.0, min(2.0, newIntensity)))
        default:
            break
        }
    }
    
    private func handleAIAssistantGesture(_ gesture: ARGestureData) {
        guard let assistant = aiAssistant else { return }
        
        switch gesture.type {
        case .doubleTap:
            assistant.startListening()
        case .longPress:
            if let worldPos = gesture.worldPosition {
                assistant.pointAtLocation(worldPos)
            }
        default:
            break
        }
    }
    
    private func findNearestClause(to position: SIMD3<Float>, in document: AnalyzedDocument) -> ClauseData? {
        var nearestClause: ClauseData?
        var minDistance: Float = Float.greatestFiniteMagnitude
        
        for clause in document.clauses {
            let clausePosition = calculateClausePosition(clause)
            let distance = simd_distance(position, clausePosition)
            
            if distance < minDistance {
                minDistance = distance
                nearestClause = clause
            }
        }
        
        return minDistance < 0.5 ? nearestClause : nil
    }
}

extension ARCoordinator: ARVoiceDelegate {
    public func didReceiveVoiceCommand(_ command: VoiceCommand) {
        analytics.trackVoiceCommand(command.intent)
        
        switch command.intent {
        case .scan:
            if !activeFeatures.contains(.documentScanning) {
                activateFeature(.documentScanning)
            }
        case .analyze:
            if let document = currentDocument {
                aiAssistant?.suggestImprovements()
            }
        case .explain:
            aiAssistant?.enableVisualSearch()
        case .highlight:
            if !activeFeatures.contains(.riskHeatmap) {
                activateFeature(.riskHeatmap)
            }
        case .translate:
            aiAssistant?.enableTranslationOverlay()
        case .help:
            aiAssistant?.startListening()
        default:
            break
        }
    }
}

extension ARCoordinator: ARHandTrackingDelegate {
    public func didUpdateHandPoses(_ poses: [HandPose]) {
        for pose in poses {
            analytics.trackHandGesture(pose.gesture)
            
            switch pose.gesture {
            case .pointing:
                if let indexTip = pose.keyPoints.first(where: { $0.joint == .indexTip }) {
                    gestureController?.performHandPoint(at: indexTip.worldPosition)
                }
            case .pinch:
                // Handle pinch gesture for selection
                if let thumbTip = pose.keyPoints.first(where: { $0.joint == .thumbTip }) {
                    gestureController?.performAirTap(at: thumbTip.worldPosition)
                }
            default:
                break
            }
        }
    }
}

// MARK: - AR Features Enumeration

public enum ARFeature: String, CaseIterable {
    case documentScanning = "documentScanning"
    case clauseVisualization = "clauseVisualization"
    case riskHeatmap = "riskHeatmap"
    case collaboration = "collaboration"
    case aiAssistant = "aiAssistant"
    case gestureControl = "gestureControl"
    
    public var displayName: String {
        switch self {
        case .documentScanning:
            return "Document Scanning"
        case .clauseVisualization:
            return "Clause Visualization"
        case .riskHeatmap:
            return "Risk Heatmap"
        case .collaboration:
            return "Collaboration"
        case .aiAssistant:
            return "AI Assistant"
        case .gestureControl:
            return "Gesture Control"
        }
    }
    
    public var description: String {
        switch self {
        case .documentScanning:
            return "Scan and digitize documents using AR"
        case .clauseVisualization:
            return "Visualize document clauses in 3D space"
        case .riskHeatmap:
            return "Show risk levels as color-coded overlays"
        case .collaboration:
            return "Share AR sessions with other users"
        case .aiAssistant:
            return "Get AI-powered help and explanations"
        case .gestureControl:
            return "Control AR features with gestures and voice"
        }
    }
}

// MARK: - Extension for GestureController Analytics

extension ARGestureController {
    func getSessionAnalytics() -> GestureAnalyticsReport? {
        // This would be implemented in the actual GestureController
        // For now, return nil as placeholder
        return nil
    }
}