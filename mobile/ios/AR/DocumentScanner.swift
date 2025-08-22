//
//  DocumentScanner.swift
//  AR Document Analysis Framework
//
//  Advanced AR document scanning with real-time detection and OCR
//

import ARKit
import Vision
import RealityKit
import AVFoundation
import Combine

@available(iOS 13.0, *)
public class DocumentScanner: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isScanning = false
    @Published public var scannedDocuments: [ScannedDocument] = []
    @Published public var currentDocument: ScannedDocument?
    @Published public var scanningStatus: ScanningStatus = .idle
    @Published public var ocrProgress: Float = 0.0
    
    private var arView: ARView?
    private var session: ARSession?
    private let visionQueue = DispatchQueue(label: "com.documentanalyzer.vision", qos: .userInteractive)
    private let ocrQueue = DispatchQueue(label: "com.documentanalyzer.ocr", qos: .userInitiated)
    
    // Vision framework
    private lazy var documentDetectionRequest: VNDetectDocumentSegmentationRequest = {
        let request = VNDetectDocumentSegmentationRequest()
        request.preferBackgroundProcessing = false
        return request
    }()
    
    private lazy var rectangleDetectionRequest: VNDetectRectanglesRequest = {
        let request = VNDetectRectanglesRequest()
        request.maximumObservations = 10
        request.minimumAspectRatio = 0.3
        request.maximumAspectRatio = 1.7
        request.minimumSize = 0.1
        request.minimumConfidence = 0.6
        return request
    }()
    
    private lazy var textRecognitionRequest: VNRecognizeTextRequest = {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US", "es-ES", "fr-FR", "de-DE", "zh-Hans", "ja-JP"]
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true
        return request
    }()
    
    // AR elements
    private var documentAnchor: AnchorEntity?
    private var scanningOverlay: Entity?
    private var gestureRecognizer: UITapGestureRecognizer?
    private var speechRecognizer: SpeechRecognitionManager?
    
    // Configuration
    public var scanningConfiguration = ScanningConfiguration()
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupVisionRequests()
        speechRecognizer = SpeechRecognitionManager()
    }
    
    // MARK: - Public Methods
    
    public func startScanning(in arView: ARView) {
        self.arView = arView
        self.session = arView.session
        
        guard ARWorldTrackingConfiguration.isSupported else {
            scanningStatus = .error("AR not supported on this device")
            return
        }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        
        if ARWorldTrackingConfiguration.supportsUserFaceTracking {
            configuration.userFaceTrackingEnabled = true
        }
        
        arView.session.run(configuration, options: [.removeExistingAnchors, .resetTracking])
        
        setupGestureRecognition(in: arView)
        setupVoiceCommands()
        
        isScanning = true
        scanningStatus = .scanning
        
        startContinuousDocumentDetection()
    }
    
    public func stopScanning() {
        isScanning = false
        scanningStatus = .idle
        session?.pause()
        
        removeScanningOverlay()
        gestureRecognizer = nil
        speechRecognizer?.stopListening()
    }
    
    public func captureDocument(at location: CGPoint) async -> ScannedDocument? {
        guard let arView = arView,
              let frame = arView.session.currentFrame else { return nil }
        
        scanningStatus = .capturing
        
        let results = await performDocumentDetection(on: frame.capturedImage, at: location)
        
        if let detectedDocument = results {
            let scannedDoc = await processScannedDocument(detectedDocument, from: frame)
            scannedDocuments.append(scannedDoc)
            currentDocument = scannedDoc
            
            await createARVisualization(for: scannedDoc, in: arView)
            scanningStatus = .completed
            
            return scannedDoc
        } else {
            scanningStatus = .error("No document detected")
            return nil
        }
    }
    
    public func captureMultiPageDocument() async -> [ScannedDocument] {
        var pages: [ScannedDocument] = []
        scanningStatus = .multiPageScanning
        
        // Implement multi-page scanning logic
        // This would involve continuous capture with page detection
        
        return pages
    }
    
    public func retryOCR(for document: ScannedDocument) async -> ScannedDocument {
        scanningStatus = .processing
        ocrProgress = 0.0
        
        let updatedDocument = await performOCR(on: document)
        
        if let index = scannedDocuments.firstIndex(where: { $0.id == document.id }) {
            scannedDocuments[index] = updatedDocument
        }
        
        scanningStatus = .completed
        return updatedDocument
    }
    
    // MARK: - Private Methods
    
    private func setupVisionRequests() {
        documentDetectionRequest.completionHandler = { [weak self] request, error in
            self?.handleDocumentDetection(request: request, error: error)
        }
        
        rectangleDetectionRequest.completionHandler = { [weak self] request, error in
            self?.handleRectangleDetection(request: request, error: error)
        }
        
        textRecognitionRequest.completionHandler = { [weak self] request, error in
            self?.handleTextRecognition(request: request, error: error)
        }
    }
    
    private func startContinuousDocumentDetection() {
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] timer in
            guard let self = self, self.isScanning else {
                timer.invalidate()
                return
            }
            
            self.performContinuousDetection()
        }
    }
    
    private func performContinuousDetection() {
        guard let arView = arView,
              let frame = arView.session.currentFrame else { return }
        
        visionQueue.async { [weak self] in
            let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .up, options: [:])
            
            do {
                try handler.perform([self?.documentDetectionRequest, self?.rectangleDetectionRequest].compactMap { $0 })
            } catch {
                DispatchQueue.main.async {
                    self?.scanningStatus = .error("Detection failed: \(error.localizedDescription)")
                }
            }
        }
    }
    
    private func performDocumentDetection(on pixelBuffer: CVPixelBuffer, at location: CGPoint) async -> DetectedDocument? {
        return await withCheckedContinuation { continuation in
            visionQueue.async {
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
                
                let request = VNDetectDocumentSegmentationRequest { request, error in
                    if let error = error {
                        print("Document detection error: \(error)")
                        continuation.resume(returning: nil)
                        return
                    }
                    
                    guard let observations = request.results as? [VNDocumentObservation],
                          let bestObservation = observations.first else {
                        continuation.resume(returning: nil)
                        return
                    }
                    
                    let detectedDoc = DetectedDocument(
                        boundingBox: bestObservation.boundingBox,
                        confidence: bestObservation.confidence,
                        pixelBuffer: pixelBuffer
                    )
                    
                    continuation.resume(returning: detectedDoc)
                }
                
                do {
                    try handler.perform([request])
                } catch {
                    print("Handler error: \(error)")
                    continuation.resume(returning: nil)
                }
            }
        }
    }
    
    private func processScannedDocument(_ detected: DetectedDocument, from frame: ARFrame) async -> ScannedDocument {
        let correctedImage = await performPerspectiveCorrection(detected)
        let ocrResult = await performOCR(on: detected)
        
        let scannedDoc = ScannedDocument(
            id: UUID(),
            originalImage: detected.pixelBuffer,
            correctedImage: correctedImage,
            boundingBox: detected.boundingBox,
            confidence: detected.confidence,
            ocrResult: ocrResult,
            timestamp: Date(),
            cameraTransform: frame.camera.transform,
            worldMappingStatus: frame.worldMappingStatus
        )
        
        return scannedDoc
    }
    
    private func performPerspectiveCorrection(_ detected: DetectedDocument) async -> UIImage? {
        return await withCheckedContinuation { continuation in
            visionQueue.async {
                // Convert CVPixelBuffer to CIImage
                let ciImage = CIImage(cvPixelBuffer: detected.pixelBuffer)
                
                // Apply perspective correction based on detected corners
                let filter = CIFilter.perspectiveCorrection()
                filter.inputImage = ciImage
                
                // Set corner points from bounding box
                let boundingBox = detected.boundingBox
                let imageSize = ciImage.extent.size
                
                filter.topLeft = CGPoint(
                    x: boundingBox.minX * imageSize.width,
                    y: (1 - boundingBox.maxY) * imageSize.height
                )
                filter.topRight = CGPoint(
                    x: boundingBox.maxX * imageSize.width,
                    y: (1 - boundingBox.maxY) * imageSize.height
                )
                filter.bottomLeft = CGPoint(
                    x: boundingBox.minX * imageSize.width,
                    y: (1 - boundingBox.minY) * imageSize.height
                )
                filter.bottomRight = CGPoint(
                    x: boundingBox.maxX * imageSize.width,
                    y: (1 - boundingBox.minY) * imageSize.height
                )
                
                guard let outputImage = filter.outputImage else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let context = CIContext()
                guard let cgImage = context.createCGImage(outputImage, from: outputImage.extent) else {
                    continuation.resume(returning: nil)
                    return
                }
                
                let correctedImage = UIImage(cgImage: cgImage)
                continuation.resume(returning: correctedImage)
            }
        }
    }
    
    private func performOCR(on document: ScannedDocument) async -> OCRResult {
        return await performOCR(on: DetectedDocument(
            boundingBox: document.boundingBox,
            confidence: document.confidence,
            pixelBuffer: document.originalImage
        ))
    }
    
    private func performOCR(on detected: DetectedDocument) async -> OCRResult {
        return await withCheckedContinuation { continuation in
            ocrQueue.async { [weak self] in
                let handler = VNImageRequestHandler(cvPixelBuffer: detected.pixelBuffer, orientation: .up, options: [:])
                
                let request = VNRecognizeTextRequest { request, error in
                    DispatchQueue.main.async {
                        self?.ocrProgress = 1.0
                    }
                    
                    if let error = error {
                        let result = OCRResult(
                            text: "",
                            confidence: 0.0,
                            language: "unknown",
                            blocks: [],
                            error: error.localizedDescription
                        )
                        continuation.resume(returning: result)
                        return
                    }
                    
                    guard let observations = request.results as? [VNRecognizedTextObservation] else {
                        let result = OCRResult(
                            text: "",
                            confidence: 0.0,
                            language: "unknown",
                            blocks: [],
                            error: "No text detected"
                        )
                        continuation.resume(returning: result)
                        return
                    }
                    
                    var fullText = ""
                    var totalConfidence: Float = 0.0
                    var blocks: [TextBlock] = []
                    
                    for observation in observations {
                        guard let candidate = observation.topCandidates(1).first else { continue }
                        
                        fullText += candidate.string + "\n"
                        totalConfidence += candidate.confidence
                        
                        let block = TextBlock(
                            text: candidate.string,
                            boundingBox: observation.boundingBox,
                            confidence: candidate.confidence
                        )
                        blocks.append(block)
                    }
                    
                    let avgConfidence = observations.isEmpty ? 0.0 : totalConfidence / Float(observations.count)
                    let detectedLanguage = self?.detectLanguage(from: fullText) ?? "en"
                    
                    let result = OCRResult(
                        text: fullText.trimmingCharacters(in: .whitespacesAndNewlines),
                        confidence: avgConfidence,
                        language: detectedLanguage,
                        blocks: blocks,
                        error: nil
                    )
                    
                    continuation.resume(returning: result)
                }
                
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = true
                request.automaticallyDetectsLanguage = true
                
                // Progress tracking
                DispatchQueue.main.async {
                    self?.ocrProgress = 0.5
                }
                
                do {
                    try handler.perform([request])
                } catch {
                    let result = OCRResult(
                        text: "",
                        confidence: 0.0,
                        language: "unknown",
                        blocks: [],
                        error: error.localizedDescription
                    )
                    continuation.resume(returning: result)
                }
            }
        }
    }
    
    private func createARVisualization(for document: ScannedDocument, in arView: ARView) async {
        await MainActor.run {
            // Remove existing visualization
            removeScanningOverlay()
            
            // Create anchor for the document
            let anchorEntity = AnchorEntity(world: document.cameraTransform)
            
            // Create document plane
            let mesh = MeshResource.generatePlane(width: 0.3, depth: 0.2)
            var material = SimpleMaterial(color: .white, isMetallic: false)
            
            if let correctedImage = document.correctedImage,
               let texture = try? TextureResource.generate(from: correctedImage, options: .init(semantic: .color)) {
                material.baseColor = MaterialColorParameter.texture(texture)
            }
            
            let documentEntity = ModelEntity(mesh: mesh, materials: [material])
            documentEntity.position = [0, 0, 0]
            
            // Add scanning indicator
            let scanIndicator = createScanningIndicator()
            scanIndicator.position = [0, 0.1, 0]
            
            anchorEntity.addChild(documentEntity)
            anchorEntity.addChild(scanIndicator)
            
            arView.scene.addAnchor(anchorEntity)
            self.documentAnchor = anchorEntity
        }
    }
    
    private func createScanningIndicator() -> Entity {
        let mesh = MeshResource.generateSphere(radius: 0.02)
        let material = SimpleMaterial(color: .systemGreen, isMetallic: false)
        let entity = ModelEntity(mesh: mesh, materials: [material])
        
        // Add pulsing animation
        let animation = AnimationResource.makeTransform(
            duration: 1.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1, 1, 1), translation: entity.position)),
                .init(time: 0.5, value: Transform(scale: SIMD3<Float>(1.2, 1.2, 1.2), translation: entity.position)),
                .init(time: 1, value: Transform(scale: SIMD3<Float>(1, 1, 1), translation: entity.position))
            ]
        )
        
        entity.playAnimation(animation)
        
        return entity
    }
    
    private func removeScanningOverlay() {
        documentAnchor?.removeFromParent()
        documentAnchor = nil
    }
    
    private func setupGestureRecognition(in arView: ARView) {
        gestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        arView.addGestureRecognizer(gestureRecognizer!)
        
        // Add pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        arView.addGestureRecognizer(pinchGesture)
    }
    
    private func setupVoiceCommands() {
        speechRecognizer?.delegate = self
        speechRecognizer?.startListening()
    }
    
    private func detectLanguage(from text: String) -> String {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        return recognizer.dominantLanguage?.rawValue ?? "en"
    }
    
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let arView = arView, isScanning else { return }
        
        let location = gesture.location(in: arView)
        
        Task {
            await captureDocument(at: location)
        }
    }
    
    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        // Implement zoom functionality for detailed document inspection
        guard let documentAnchor = documentAnchor else { return }
        
        if gesture.state == .changed {
            let scale = Float(gesture.scale)
            documentAnchor.transform.scale = SIMD3<Float>(scale, scale, 1.0)
        }
    }
    
    // MARK: - Vision Request Handlers
    
    private func handleDocumentDetection(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNDocumentObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            self?.updateDocumentOverlay(with: observations)
        }
    }
    
    private func handleRectangleDetection(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNRectangleObservation] else { return }
        
        DispatchQueue.main.async { [weak self] in
            self?.updateRectangleOverlay(with: observations)
        }
    }
    
    private func handleTextRecognition(request: VNRequest, error: Error?) {
        // Handle continuous text recognition results if needed
    }
    
    private func updateDocumentOverlay(with observations: [VNDocumentObservation]) {
        guard let arView = arView, !observations.isEmpty else { return }
        
        // Update AR overlay to show detected documents
        // This would involve updating the visual indicators in the AR scene
    }
    
    private func updateRectangleOverlay(with observations: [VNRectangleObservation]) {
        // Update rectangle detection overlay
    }
}

// MARK: - SpeechRecognitionDelegate

extension DocumentScanner: SpeechRecognitionDelegate {
    func speechRecognizer(_ recognizer: SpeechRecognitionManager, didRecognize text: String) {
        handleVoiceCommand(text)
    }
    
    func speechRecognizer(_ recognizer: SpeechRecognitionManager, didFailWithError error: Error) {
        print("Speech recognition error: \(error)")
    }
    
    private func handleVoiceCommand(_ command: String) {
        let lowercaseCommand = command.lowercased()
        
        if lowercaseCommand.contains("scan") || lowercaseCommand.contains("capture") {
            Task {
                guard let arView = arView else { return }
                let center = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
                await captureDocument(at: center)
            }
        } else if lowercaseCommand.contains("stop") {
            stopScanning()
        } else if lowercaseCommand.contains("retry") {
            if let currentDoc = currentDocument {
                Task {
                    await retryOCR(for: currentDoc)
                }
            }
        }
    }
}

// MARK: - Supporting Types

public enum ScanningStatus {
    case idle
    case scanning
    case capturing
    case processing
    case multiPageScanning
    case completed
    case error(String)
}

public struct ScanningConfiguration {
    public var autoCapture = false
    public var multiPageMode = false
    public var ocrLanguages = ["en-US"]
    public var minimumConfidence: Float = 0.6
    public var enableVoiceCommands = true
    public var enableGestures = true
    
    public init() {}
}

public struct DetectedDocument {
    public let boundingBox: CGRect
    public let confidence: Float
    public let pixelBuffer: CVPixelBuffer
}

public struct ScannedDocument: Identifiable {
    public let id: UUID
    public let originalImage: CVPixelBuffer
    public let correctedImage: UIImage?
    public let boundingBox: CGRect
    public let confidence: Float
    public let ocrResult: OCRResult
    public let timestamp: Date
    public let cameraTransform: simd_float4x4
    public let worldMappingStatus: ARFrame.WorldMappingStatus
}

public struct OCRResult {
    public let text: String
    public let confidence: Float
    public let language: String
    public let blocks: [TextBlock]
    public let error: String?
}

public struct TextBlock {
    public let text: String
    public let boundingBox: CGRect
    public let confidence: Float
}

// MARK: - Speech Recognition Manager

private class SpeechRecognitionManager: NSObject {
    weak var delegate: SpeechRecognitionDelegate?
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    
    override init() {
        super.init()
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    }
    
    func startListening() {
        guard let speechRecognizer = speechRecognizer,
              speechRecognizer.isAvailable else { return }
        
        SFSpeechRecognizer.requestAuthorization { status in
            if status == .authorized {
                DispatchQueue.main.async {
                    self.startRecording()
                }
            }
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
    }
    
    private func startRecording() {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { return }
        
        recognitionRequest.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            if let result = result {
                let text = result.bestTranscription.formattedString
                self?.delegate?.speechRecognizer(self!, didRecognize: text)
            }
            
            if let error = error {
                self?.delegate?.speechRecognizer(self!, didFailWithError: error)
            }
        }
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            recognitionRequest.append(buffer)
        }
        
        audioEngine.prepare()
        
        do {
            try audioEngine.start()
        } catch {
            delegate?.speechRecognizer(self, didFailWithError: error)
        }
    }
}

private protocol SpeechRecognitionDelegate: AnyObject {
    func speechRecognizer(_ recognizer: SpeechRecognitionManager, didRecognize text: String)
    func speechRecognizer(_ recognizer: SpeechRecognitionManager, didFailWithError error: Error)
}