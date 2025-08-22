//
//  ARGestureController.swift
//  AR Document Analysis Framework
//
//  Advanced gesture and voice control systems for AR interactions
//

import ARKit
import RealityKit
import SwiftUI
import Combine
import Speech
import AVFoundation
import Vision

@available(iOS 13.0, *)
public class ARGestureController: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isGestureRecognitionActive = false
    @Published public var isVoiceControlActive = false
    @Published public var isHandTrackingActive = false
    @Published public var currentGesture: ARGestureType?
    @Published public var gestureConfidence: Float = 0.0
    @Published public var voiceCommandHistory: [VoiceCommand] = []
    @Published public var handPoses: [HandPose] = []
    
    private var arView: ARView?
    private var gestureRecognizers: [UIGestureRecognizer] = []
    private var handTrackingSession: VNSequenceRequestHandler?
    
    // Voice recognition
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var audioEngine = AVAudioEngine()
    
    // Hand tracking
    private var handPoseRequest: VNDetectHumanHandPoseRequest?
    private var handTrackingTimer: Timer?
    
    // Gesture delegates
    public weak var gestureDelegate: ARGestureDelegate?
    public weak var voiceDelegate: ARVoiceDelegate?
    public weak var handTrackingDelegate: ARHandTrackingDelegate?
    
    // Gesture configuration
    private var gestureSettings = ARGestureSettings()
    
    // Command processing
    private var commandProcessor = VoiceCommandProcessor()
    private var gestureHistory: [ARGestureData] = []
    private let maxGestureHistory = 50
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupSpeechRecognition()
        setupHandTracking()
    }
    
    deinit {
        deactivateAllControls()
    }
    
    // MARK: - Public Methods
    
    public func activateGestureRecognition(in arView: ARView, settings: ARGestureSettings = ARGestureSettings()) {
        self.arView = arView
        self.gestureSettings = settings
        
        setupGestureRecognizers()
        isGestureRecognitionActive = true
        
        ARDebugUtils.logEvent("Gesture recognition activated", data: [
            "enabledGestures": settings.enabledGestures.map { $0.rawValue }
        ])
    }
    
    public func activateVoiceControl() {
        guard !isVoiceControlActive else { return }
        
        requestSpeechAuthorization { [weak self] authorized in
            if authorized {
                DispatchQueue.main.async {
                    self?.startVoiceRecognition()
                    self?.isVoiceControlActive = true
                    ARDebugUtils.logEvent("Voice control activated")
                }
            }
        }
    }
    
    public func activateHandTracking() {
        guard !isHandTrackingActive else { return }
        
        setupHandTrackingSession()
        startHandTracking()
        isHandTrackingActive = true
        
        ARDebugUtils.logEvent("Hand tracking activated")
    }
    
    public func deactivateAllControls() {
        deactivateGestureRecognition()
        deactivateVoiceControl()
        deactivateHandTracking()
    }
    
    public func deactivateGestureRecognition() {
        removeGestureRecognizers()
        isGestureRecognitionActive = false
        ARDebugUtils.logEvent("Gesture recognition deactivated")
    }
    
    public func deactivateVoiceControl() {
        stopVoiceRecognition()
        isVoiceControlActive = false
        ARDebugUtils.logEvent("Voice control deactivated")
    }
    
    public func deactivateHandTracking() {
        stopHandTracking()
        isHandTrackingActive = false
        ARDebugUtils.logEvent("Hand tracking deactivated")
    }
    
    public func processVoiceCommand(_ command: String) {
        let voiceCommand = commandProcessor.processCommand(command)
        voiceCommandHistory.append(voiceCommand)
        
        // Keep only last 20 commands
        if voiceCommandHistory.count > 20 {
            voiceCommandHistory.removeFirst()
        }
        
        voiceDelegate?.didReceiveVoiceCommand(voiceCommand)
        ARDebugUtils.logEvent("Voice command processed", data: [
            "command": command,
            "intent": voiceCommand.intent.rawValue,
            "confidence": voiceCommand.confidence
        ])
    }
    
    public func updateGestureSettings(_ settings: ARGestureSettings) {
        self.gestureSettings = settings
        
        if isGestureRecognitionActive {
            removeGestureRecognizers()
            setupGestureRecognizers()
        }
    }
    
    public func performAirTap(at position: SIMD3<Float>) {
        let gestureData = ARGestureData(
            type: .airTap,
            location: .zero,
            worldPosition: position,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    public func performHandPoint(at position: SIMD3<Float>, duration: TimeInterval = 1.0) {
        let gestureData = ARGestureData(
            type: .handPoint,
            location: .zero,
            worldPosition: position,
            timestamp: Date()
        )
        
        processGesture(gestureData)
        
        // Auto-stop after duration
        DispatchQueue.main.asyncAfter(deadline: .now() + duration) {
            self.currentGesture = nil
        }
    }
    
    // MARK: - Private Methods
    
    private func setupSpeechRecognition() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        speechRecognizer?.delegate = self
    }
    
    private func setupHandTracking() {
        handPoseRequest = VNDetectHumanHandPoseRequest()
        handPoseRequest?.maximumHandCount = 2
        handTrackingSession = VNSequenceRequestHandler()
    }
    
    private func setupGestureRecognizers() {
        guard let arView = arView else { return }
        
        removeGestureRecognizers()
        
        for gestureType in gestureSettings.enabledGestures {
            switch gestureType {
            case .tap:
                setupTapGesture(in: arView)
            case .doubleTap:
                setupDoubleTapGesture(in: arView)
            case .longPress:
                setupLongPressGesture(in: arView)
            case .pinch:
                setupPinchGesture(in: arView)
            case .rotation:
                setupRotationGesture(in: arView)
            case .pan:
                setupPanGesture(in: arView)
            case .swipe:
                setupSwipeGesture(in: arView)
            default:
                break // Air gestures handled separately
            }
        }
    }
    
    private func setupTapGesture(in arView: ARView) {
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tapGesture.numberOfTapsRequired = 1
        arView.addGestureRecognizer(tapGesture)
        gestureRecognizers.append(tapGesture)
    }
    
    private func setupDoubleTapGesture(in arView: ARView) {
        let doubleTapGesture = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        arView.addGestureRecognizer(doubleTapGesture)
        gestureRecognizers.append(doubleTapGesture)
    }
    
    private func setupLongPressGesture(in arView: ARView) {
        let longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        longPressGesture.minimumPressDuration = 0.8
        arView.addGestureRecognizer(longPressGesture)
        gestureRecognizers.append(longPressGesture)
    }
    
    private func setupPinchGesture(in arView: ARView) {
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        arView.addGestureRecognizer(pinchGesture)
        gestureRecognizers.append(pinchGesture)
    }
    
    private func setupRotationGesture(in arView: ARView) {
        let rotationGesture = UIRotationGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
        arView.addGestureRecognizer(rotationGesture)
        gestureRecognizers.append(rotationGesture)
    }
    
    private func setupPanGesture(in arView: ARView) {
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.minimumNumberOfTouches = 1
        panGesture.maximumNumberOfTouches = 1
        arView.addGestureRecognizer(panGesture)
        gestureRecognizers.append(panGesture)
    }
    
    private func setupSwipeGesture(in arView: ARView) {
        for direction in [UISwipeGestureRecognizer.Direction.up, .down, .left, .right] {
            let swipeGesture = UISwipeGestureRecognizer(target: self, action: #selector(handleSwipe(_:)))
            swipeGesture.direction = direction
            arView.addGestureRecognizer(swipeGesture)
            gestureRecognizers.append(swipeGesture)
        }
    }
    
    private func removeGestureRecognizers() {
        guard let arView = arView else { return }
        
        for recognizer in gestureRecognizers {
            arView.removeGestureRecognizer(recognizer)
        }
        gestureRecognizers.removeAll()
    }
    
    // MARK: - Gesture Handlers
    
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard gesture.state == .ended else { return }
        
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        
        let gestureData = ARGestureData(
            type: .tap,
            location: location,
            worldPosition: worldPosition,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    @objc private func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        guard gesture.state == .ended else { return }
        
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        
        let gestureData = ARGestureData(
            type: .doubleTap,
            location: location,
            worldPosition: worldPosition,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    @objc private func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else { return }
        
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        
        let gestureData = ARGestureData(
            type: .longPress,
            location: location,
            worldPosition: worldPosition,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        let velocity = gesture.velocity
        
        let gestureData = ARGestureData(
            type: .pinch,
            location: location,
            worldPosition: worldPosition,
            velocity: CGPoint(x: velocity, y: 0),
            scale: Float(gesture.scale),
            timestamp: Date()
        )
        
        processGesture(gestureData)
        
        gesture.scale = 1.0 // Reset scale for next gesture
    }
    
    @objc private func handleRotation(_ gesture: UIRotationGestureRecognizer) {
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        let velocity = gesture.velocity
        
        let gestureData = ARGestureData(
            type: .rotation,
            location: location,
            worldPosition: worldPosition,
            velocity: CGPoint(x: velocity, y: 0),
            rotation: Float(gesture.rotation),
            timestamp: Date()
        )
        
        processGesture(gestureData)
        
        gesture.rotation = 0.0 // Reset rotation for next gesture
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        let velocity = gesture.velocity(in: gesture.view!)
        
        let gestureData = ARGestureData(
            type: .pan,
            location: location,
            worldPosition: worldPosition,
            velocity: velocity,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    @objc private func handleSwipe(_ gesture: UISwipeGestureRecognizer) {
        guard gesture.state == .ended else { return }
        
        let location = gesture.location(in: gesture.view!)
        let worldPosition = getWorldPosition(for: location)
        
        // Determine velocity based on direction
        var velocity = CGPoint.zero
        switch gesture.direction {
        case .up:
            velocity = CGPoint(x: 0, y: -500)
        case .down:
            velocity = CGPoint(x: 0, y: 500)
        case .left:
            velocity = CGPoint(x: -500, y: 0)
        case .right:
            velocity = CGPoint(x: 500, y: 0)
        default:
            break
        }
        
        let gestureData = ARGestureData(
            type: .swipe,
            location: location,
            worldPosition: worldPosition,
            velocity: velocity,
            timestamp: Date()
        )
        
        processGesture(gestureData)
    }
    
    private func getWorldPosition(for location: CGPoint) -> SIMD3<Float>? {
        guard let arView = arView else { return nil }
        
        // Try to get world position through ray casting
        let raycastResults = arView.raycast(from: location, allowing: .estimatedPlane, alignment: .any)
        
        if let firstResult = raycastResults.first {
            return SIMD3<Float>(firstResult.worldTransform.columns.3.x,
                               firstResult.worldTransform.columns.3.y,
                               firstResult.worldTransform.columns.3.z)
        }
        
        return nil
    }
    
    private func processGesture(_ gestureData: ARGestureData) {
        // Add to history
        gestureHistory.append(gestureData)
        if gestureHistory.count > maxGestureHistory {
            gestureHistory.removeFirst()
        }
        
        // Update current state
        currentGesture = gestureData.type
        gestureConfidence = calculateGestureConfidence(gestureData)
        
        // Notify delegate
        gestureDelegate?.didReceiveGesture(gestureData)
        
        // Auto-clear current gesture after delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            if self.currentGesture == gestureData.type {
                self.currentGesture = nil
            }
        }
    }
    
    private func calculateGestureConfidence(_ gestureData: ARGestureData) -> Float {
        // Calculate confidence based on gesture characteristics
        var confidence: Float = 1.0
        
        // Reduce confidence if no world position available
        if gestureData.worldPosition == nil {
            confidence *= 0.7
        }
        
        // Adjust based on velocity for moving gestures
        if gestureData.type == .pan || gestureData.type == .swipe {
            let velocityMagnitude = sqrt(pow(gestureData.velocity.x, 2) + pow(gestureData.velocity.y, 2))
            confidence *= min(1.0, Float(velocityMagnitude / 1000.0))
        }
        
        return confidence
    }
    
    // MARK: - Voice Recognition
    
    private func requestSpeechAuthorization(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                completion(status == .authorized)
            }
        }
    }
    
    private func startVoiceRecognition() {
        guard let speechRecognizer = speechRecognizer,
              speechRecognizer.isAvailable else { return }
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionRequest.shouldReportPartialResults = true
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
            
            recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                if let result = result {
                    let transcribedText = result.bestTranscription.formattedString
                    
                    if result.isFinal {
                        DispatchQueue.main.async {
                            self?.processVoiceCommand(transcribedText)
                        }
                    }
                }
                
                if let error = error {
                    print("Speech recognition error: \(error)")
                }
            }
            
        } catch {
            print("Audio engine start error: \(error)")
        }
    }
    
    private func stopVoiceRecognition() {
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
    }
    
    // MARK: - Hand Tracking
    
    private func setupHandTrackingSession() {
        guard let arView = arView else { return }
        
        // Enable hand tracking in AR configuration if supported
        if let configuration = arView.session.configuration as? ARWorldTrackingConfiguration {
            // Hand tracking would be configured here if available
        }
    }
    
    private func startHandTracking() {
        handTrackingTimer = Timer.scheduledTimer(withTimeInterval: 1/30.0, repeats: true) { [weak self] _ in
            self?.processHandTracking()
        }
    }
    
    private func stopHandTracking() {
        handTrackingTimer?.invalidate()
        handTrackingTimer = nil
    }
    
    private func processHandTracking() {
        guard let arView = arView,
              let frame = arView.session.currentFrame,
              let handPoseRequest = handPoseRequest,
              let handTrackingSession = handTrackingSession else { return }
        
        let pixelBuffer = frame.capturedImage
        
        do {
            try handTrackingSession.perform([handPoseRequest], on: pixelBuffer, orientation: .up)
            
            guard let observations = handPoseRequest.results else { return }
            
            var detectedPoses: [HandPose] = []
            
            for observation in observations {
                if let handPose = processHandObservation(observation, in: frame) {
                    detectedPoses.append(handPose)
                }
            }
            
            DispatchQueue.main.async {
                self.handPoses = detectedPoses
                self.handTrackingDelegate?.didUpdateHandPoses(detectedPoses)
            }
            
        } catch {
            print("Hand tracking error: \(error)")
        }
    }
    
    private func processHandObservation(_ observation: VNHumanHandPoseObservation, in frame: ARFrame) -> HandPose? {
        do {
            let handLandmarks = try observation.recognizedPoints(.all)
            
            // Extract key points
            var keyPoints: [HandKeyPoint] = []
            
            for (jointName, point) in handLandmarks {
                if point.confidence > 0.5 {
                    let worldPosition = convertToWorldPosition(point.location, in: frame)
                    let keyPoint = HandKeyPoint(
                        joint: jointName,
                        position: point.location,
                        worldPosition: worldPosition,
                        confidence: point.confidence
                    )
                    keyPoints.append(keyPoint)
                }
            }
            
            // Determine hand gesture
            let gestureType = recognizeHandGesture(from: keyPoints)
            
            return HandPose(
                handedness: observation.chirality == .left ? .left : .right,
                keyPoints: keyPoints,
                gesture: gestureType,
                confidence: observation.confidence,
                boundingBox: observation.boundingBox
            )
            
        } catch {
            print("Hand pose processing error: \(error)")
            return nil
        }
    }
    
    private func convertToWorldPosition(_ normalizedPoint: CGPoint, in frame: ARFrame) -> SIMD3<Float> {
        // Convert normalized point to world coordinates
        // This is a simplified implementation - full version would use proper coordinate transformation
        let viewportSize = frame.capturedImage.width
        let screenPoint = CGPoint(
            x: normalizedPoint.x * CGFloat(viewportSize),
            y: normalizedPoint.y * CGFloat(viewportSize)
        )
        
        // Use raycast to get world position
        if let arView = arView {
            let raycastResults = arView.raycast(from: screenPoint, allowing: .estimatedPlane, alignment: .any)
            if let firstResult = raycastResults.first {
                return SIMD3<Float>(firstResult.worldTransform.columns.3.x,
                                   firstResult.worldTransform.columns.3.y,
                                   firstResult.worldTransform.columns.3.z)
            }
        }
        
        // Fallback to camera-relative position
        return SIMD3<Float>(Float(normalizedPoint.x - 0.5) * 2.0, Float(0.5 - normalizedPoint.y) * 2.0, -1.0)
    }
    
    private func recognizeHandGesture(from keyPoints: [HandKeyPoint]) -> HandGestureType {
        // Simplified gesture recognition - full implementation would use ML models
        guard keyPoints.count >= 5 else { return .unknown }
        
        // Check for pointing gesture
        if isPointingGesture(keyPoints) {
            return .pointing
        }
        
        // Check for pinch gesture
        if isPinchGesture(keyPoints) {
            return .pinch
        }
        
        // Check for open palm
        if isOpenPalmGesture(keyPoints) {
            return .openPalm
        }
        
        // Check for fist
        if isFistGesture(keyPoints) {
            return .fist
        }
        
        return .unknown
    }
    
    private func isPointingGesture(_ keyPoints: [HandKeyPoint]) -> Bool {
        // Simplified pointing detection
        let indexPoints = keyPoints.filter { $0.joint.rawValue.contains("index") }
        let thumbPoints = keyPoints.filter { $0.joint.rawValue.contains("thumb") }
        
        return indexPoints.count >= 2 && thumbPoints.count >= 2
    }
    
    private func isPinchGesture(_ keyPoints: [HandKeyPoint]) -> Bool {
        // Simplified pinch detection
        let thumbTip = keyPoints.first { $0.joint == .thumbTip }
        let indexTip = keyPoints.first { $0.joint == .indexTip }
        
        guard let thumb = thumbTip, let index = indexTip else { return false }
        
        let distance = sqrt(pow(thumb.position.x - index.position.x, 2) + pow(thumb.position.y - index.position.y, 2))
        return distance < 0.05 // Close together
    }
    
    private func isOpenPalmGesture(_ keyPoints: [HandKeyPoint]) -> Bool {
        // Simplified open palm detection
        let fingerTips = keyPoints.filter { 
            [$0.joint].contains { joint in
                [.thumbTip, .indexTip, .middleTip, .ringTip, .littleTip].contains(joint)
            }
        }
        
        return fingerTips.count >= 4
    }
    
    private func isFistGesture(_ keyPoints: [HandKeyPoint]) -> Bool {
        // Simplified fist detection
        let fingerTips = keyPoints.filter { 
            [$0.joint].contains { joint in
                [.indexTip, .middleTip, .ringTip, .littleTip].contains(joint)
            }
        }
        
        return fingerTips.count <= 1
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension ARGestureController: SFSpeechRecognizerDelegate {
    public func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if !available && isVoiceControlActive {
            deactivateVoiceControl()
        }
    }
}

// MARK: - Supporting Types and Protocols

/// Delegate for AR gesture events
public protocol ARGestureDelegate: AnyObject {
    func didReceiveGesture(_ gesture: ARGestureData)
}

/// Delegate for voice command events
public protocol ARVoiceDelegate: AnyObject {
    func didReceiveVoiceCommand(_ command: VoiceCommand)
}

/// Delegate for hand tracking events
public protocol ARHandTrackingDelegate: AnyObject {
    func didUpdateHandPoses(_ poses: [HandPose])
}

/// AR gesture configuration settings
public struct ARGestureSettings {
    public var enabledGestures: [ARGestureType] = [.tap, .pinch, .pan]
    public var gestureThreshold: Float = 0.5
    public var voiceCommandTimeout: TimeInterval = 3.0
    public var handTrackingEnabled: Bool = true
    public var simultaneousGesturesAllowed: Bool = false
    public var hapticFeedbackEnabled: Bool = true
    
    public init() {}
}

/// Hand pose data structure
public struct HandPose {
    public let handedness: Handedness
    public let keyPoints: [HandKeyPoint]
    public let gesture: HandGestureType
    public let confidence: Float
    public let boundingBox: CGRect
    
    public init(handedness: Handedness, keyPoints: [HandKeyPoint], gesture: HandGestureType, confidence: Float, boundingBox: CGRect) {
        self.handedness = handedness
        self.keyPoints = keyPoints
        self.gesture = gesture
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
}

/// Hand key point information
public struct HandKeyPoint {
    public let joint: VNHumanHandPoseObservation.JointName
    public let position: CGPoint
    public let worldPosition: SIMD3<Float>
    public let confidence: Float
    
    public init(joint: VNHumanHandPoseObservation.JointName, position: CGPoint, worldPosition: SIMD3<Float>, confidence: Float) {
        self.joint = joint
        self.position = position
        self.worldPosition = worldPosition
        self.confidence = confidence
    }
}

/// Hand gesture types
public enum HandGestureType: String, CaseIterable {
    case pointing = "pointing"
    case pinch = "pinch"
    case openPalm = "openPalm"
    case fist = "fist"
    case thumbsUp = "thumbsUp"
    case peace = "peace"
    case unknown = "unknown"
}

/// Handedness enumeration
public enum Handedness: String, CaseIterable {
    case left = "left"
    case right = "right"
}

/// Voice command processor
private class VoiceCommandProcessor {
    
    func processCommand(_ command: String) -> VoiceCommand {
        let lowercaseCommand = command.lowercased()
        let intent = classifyIntent(lowercaseCommand)
        let entities = extractEntities(from: lowercaseCommand)
        let confidence = calculateConfidence(for: lowercaseCommand, intent: intent)
        
        return VoiceCommand(
            command: command,
            confidence: confidence,
            intent: intent,
            entities: entities
        )
    }
    
    private func classifyIntent(_ command: String) -> CommandIntent {
        // Simple intent classification - in production, use ML model
        if command.contains("scan") || command.contains("capture") {
            return .scan
        } else if command.contains("analyze") || command.contains("review") {
            return .analyze
        } else if command.contains("explain") || command.contains("tell me about") {
            return .explain
        } else if command.contains("highlight") || command.contains("show") {
            return .highlight
        } else if command.contains("compare") {
            return .compare
        } else if command.contains("translate") {
            return .translate
        } else if command.contains("save") {
            return .save
        } else if command.contains("share") {
            return .share
        } else if command.contains("navigate") || command.contains("go to") {
            return .navigate
        } else if command.contains("help") {
            return .help
        } else {
            return .unknown
        }
    }
    
    private func extractEntities(from command: String) -> [String] {
        var entities: [String] = []
        
        // Extract legal terms
        let legalTerms = ["arbitration", "liability", "termination", "warranty", "indemnification", "breach"]
        for term in legalTerms {
            if command.contains(term) {
                entities.append(term)
            }
        }
        
        // Extract risk levels
        let riskLevels = ["high", "low", "medium", "critical"]
        for level in riskLevels {
            if command.contains(level) {
                entities.append(level)
            }
        }
        
        return entities
    }
    
    private func calculateConfidence(for command: String, intent: CommandIntent) -> Float {
        var confidence: Float = 0.5
        
        // Increase confidence for recognized patterns
        switch intent {
        case .scan, .analyze, .explain:
            confidence = 0.9
        case .highlight, .compare:
            confidence = 0.8
        case .translate, .save, .share:
            confidence = 0.7
        case .navigate, .help:
            confidence = 0.8
        case .unknown:
            confidence = 0.3
        }
        
        // Adjust based on command length and clarity
        if command.count > 20 {
            confidence *= 0.9
        }
        
        if command.count < 3 {
            confidence *= 0.5
        }
        
        return min(1.0, max(0.0, confidence))
    }
}

// MARK: - Gesture Analytics

/// Gesture analytics tracker
public class ARGestureAnalytics {
    private var gestureCount: [ARGestureType: Int] = [:]
    private var voiceCommandCount: [CommandIntent: Int] = [:]
    private var handGestureCount: [HandGestureType: Int] = [:]
    private var sessionStartTime: Date?
    
    public init() {}
    
    public func startSession() {
        sessionStartTime = Date()
        resetCounters()
    }
    
    public func trackGesture(_ gestureType: ARGestureType) {
        gestureCount[gestureType, default: 0] += 1
    }
    
    public func trackVoiceCommand(_ intent: CommandIntent) {
        voiceCommandCount[intent, default: 0] += 1
    }
    
    public func trackHandGesture(_ gestureType: HandGestureType) {
        handGestureCount[gestureType, default: 0] += 1
    }
    
    public func getSessionReport() -> GestureAnalyticsReport {
        let sessionDuration = sessionStartTime?.timeIntervalSinceNow ?? 0
        
        return GestureAnalyticsReport(
            sessionDuration: abs(sessionDuration),
            gestureCount: gestureCount,
            voiceCommandCount: voiceCommandCount,
            handGestureCount: handGestureCount
        )
    }
    
    private func resetCounters() {
        gestureCount.removeAll()
        voiceCommandCount.removeAll()
        handGestureCount.removeAll()
    }
}

/// Gesture analytics report
public struct GestureAnalyticsReport {
    public let sessionDuration: TimeInterval
    public let gestureCount: [ARGestureType: Int]
    public let voiceCommandCount: [CommandIntent: Int]
    public let handGestureCount: [HandGestureType: Int]
    
    public var totalGestures: Int {
        return gestureCount.values.reduce(0, +)
    }
    
    public var totalVoiceCommands: Int {
        return voiceCommandCount.values.reduce(0, +)
    }
    
    public var totalHandGestures: Int {
        return handGestureCount.values.reduce(0, +)
    }
    
    public var mostUsedGesture: ARGestureType? {
        return gestureCount.max(by: { $0.value < $1.value })?.key
    }
    
    public var mostUsedVoiceCommand: CommandIntent? {
        return voiceCommandCount.max(by: { $0.value < $1.value })?.key
    }
}