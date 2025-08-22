//
//  RiskHeatmap.swift
//  AR Document Analysis Framework
//
//  AR risk overlay and heat mapping for document analysis
//

import ARKit
import RealityKit
import SwiftUI
import Combine
import Metal
import MetalKit

@available(iOS 13.0, *)
public class RiskHeatmap: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isHeatmapActive = false
    @Published public var heatmapIntensity: Float = 1.0
    @Published public var heatmapMode: HeatmapMode = .risk
    @Published public var temperatureThreshold: Float = 0.5
    @Published public var animationSpeed: Float = 1.0
    @Published public var showLegend = true
    
    private var arView: ARView?
    private var documentAnchor: AnchorEntity?
    private var heatmapEntity: Entity?
    private var riskIndicators: [UUID: RiskIndicatorEntity] = [:]
    private var heatmapTexture: TextureResource?
    private var animationController: AnimationPlaybackController?
    
    // Metal resources for heatmap generation
    private var device: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var heatmapComputeShader: MTLComputePipelineState?
    
    // Heatmap data
    private var currentRiskData: [RiskDataPoint] = []
    private var heatmapResolution: Int = 512
    private let maxRiskIndicators = 100
    
    // Color schemes
    private let riskColorScheme = HeatmapColorScheme(
        low: SIMD4<Float>(0.0, 1.0, 0.0, 0.6),    // Green
        medium: SIMD4<Float>(1.0, 1.0, 0.0, 0.7), // Yellow
        high: SIMD4<Float>(1.0, 0.5, 0.0, 0.8),   // Orange
        critical: SIMD4<Float>(1.0, 0.0, 0.0, 0.9) // Red
    )
    
    private let confidenceColorScheme = HeatmapColorScheme(
        low: SIMD4<Float>(0.5, 0.5, 0.5, 0.3),    // Gray
        medium: SIMD4<Float>(0.0, 0.5, 1.0, 0.5), // Blue
        high: SIMD4<Float>(0.0, 1.0, 1.0, 0.7),   // Cyan
        critical: SIMD4<Float>(1.0, 1.0, 1.0, 0.9) // White
    )
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupMetal()
    }
    
    // MARK: - Public Methods
    
    public func activateHeatmap(for document: AnalyzedDocument, in arView: ARView) {
        self.arView = arView
        
        setupDocumentAnchor(for: document)
        generateRiskData(from: document)
        createHeatmapVisualization()
        
        isHeatmapActive = true
        startHeatmapAnimations()
    }
    
    public func deactivateHeatmap() {
        stopAnimations()
        clearHeatmapVisualization()
        isHeatmapActive = false
    }
    
    public func updateHeatmapMode(_ mode: HeatmapMode) {
        heatmapMode = mode
        updateHeatmapVisualization()
    }
    
    public func updateIntensity(_ intensity: Float) {
        heatmapIntensity = max(0.0, min(2.0, intensity))
        updateHeatmapColors()
    }
    
    public func addRiskIndicator(at position: SIMD3<Float>, riskLevel: RiskLevel, animated: Bool = true) {
        let indicator = createRiskIndicator(riskLevel: riskLevel, at: position)
        
        if let anchor = documentAnchor {
            anchor.addChild(indicator)
            riskIndicators[indicator.id] = indicator
            
            if animated {
                animateIndicatorAppearance(indicator)
            }
        }
    }
    
    public func removeRiskIndicator(withId id: UUID, animated: Bool = true) {
        guard let indicator = riskIndicators[id] else { return }
        
        if animated {
            animateIndicatorDisappearance(indicator) {
                indicator.removeFromParent()
                self.riskIndicators.removeValue(forKey: id)
            }
        } else {
            indicator.removeFromParent()
            riskIndicators.removeValue(forKey: id)
        }
    }
    
    public func highlightRiskArea(center: SIMD3<Float>, radius: Float, riskLevel: RiskLevel) {
        let highlightEntity = createRiskAreaHighlight(center: center, radius: radius, riskLevel: riskLevel)
        documentAnchor?.addChild(highlightEntity)
        
        // Auto-remove after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
            highlightEntity.removeFromParent()
        }
    }
    
    public func updateRiskData(_ data: [RiskDataPoint]) {
        currentRiskData = data
        regenerateHeatmapTexture()
        updateRiskIndicators()
    }
    
    public func exportHeatmapImage() -> UIImage? {
        guard let texture = heatmapTexture else { return nil }
        
        // Convert RealityKit texture to UIImage
        // This would require additional Metal code to read texture data
        return nil // Placeholder - would implement texture-to-image conversion
    }
    
    // MARK: - Private Methods
    
    private func setupMetal() {
        device = MTLCreateSystemDefaultDevice()
        guard let device = device else {
            print("Metal not available")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        
        // Create compute shader for heatmap generation
        guard let library = device.makeDefaultLibrary() else { return }
        
        do {
            let heatmapFunction = library.makeFunction(name: "generateHeatmap")
            heatmapComputeShader = try device.makeComputePipelineState(function: heatmapFunction!)
        } catch {
            print("Failed to create compute pipeline: \(error)")
        }
    }
    
    private func setupDocumentAnchor(for document: AnalyzedDocument) {
        guard let arView = arView else { return }
        
        let cameraTransform = arView.session.currentFrame?.camera.transform ?? matrix_identity_float4x4
        let anchorEntity = AnchorEntity(world: cameraTransform)
        anchorEntity.position = [0, 0, -1.0]
        
        arView.scene.addAnchor(anchorEntity)
        self.documentAnchor = anchorEntity
    }
    
    private func generateRiskData(from document: AnalyzedDocument) {
        currentRiskData.removeAll()
        
        for clause in document.clauses {
            let dataPoint = RiskDataPoint(
                id: clause.id,
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
            currentRiskData.append(dataPoint)
        }
    }
    
    private func calculateClausePosition(_ clause: ClauseData) -> SIMD3<Float> {
        // Map clause properties to 3D position
        let sectionHash = Float(clause.section.hashValue % 360) * .pi / 180
        let orderOffset = Float(clause.order) * 0.1
        
        let x = cos(sectionHash) * (0.3 + orderOffset * 0.05)
        let y = Float(clause.riskLevel.rawValue - 1) * 0.15
        let z = sin(sectionHash) * (0.3 + orderOffset * 0.05)
        
        return SIMD3<Float>(x, y, z)
    }
    
    private func createHeatmapVisualization() {
        generateHeatmapTexture()
        createHeatmapPlane()
        createRiskIndicators()
        
        if showLegend {
            createHeatmapLegend()
        }
    }
    
    private func generateHeatmapTexture() {
        guard let device = device,
              let commandQueue = commandQueue,
              let computeShader = heatmapComputeShader else { return }
        
        // Create texture descriptor
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: heatmapResolution,
            height: heatmapResolution,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderWrite, .shaderRead]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { return }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        computeEncoder.setComputePipelineState(computeShader)
        computeEncoder.setTexture(texture, index: 0)
        
        // Set heatmap data buffer
        let riskDataBuffer = createRiskDataBuffer()
        computeEncoder.setBuffer(riskDataBuffer, offset: 0, index: 0)
        
        // Set parameters
        var params = HeatmapParameters(
            intensity: heatmapIntensity,
            threshold: temperatureThreshold,
            mode: Int32(heatmapMode.rawValue),
            dataCount: Int32(currentRiskData.count)
        )
        
        computeEncoder.setBytes(&params, length: MemoryLayout<HeatmapParameters>.size, index: 1)
        
        // Dispatch
        let threadsPerGroup = MTLSize(width: 16, height: 16, depth: 1)
        let numThreadgroups = MTLSize(
            width: (heatmapResolution + 15) / 16,
            height: (heatmapResolution + 15) / 16,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Convert to RealityKit texture
        do {
            heatmapTexture = try TextureResource.generate(from: texture, options: .init(semantic: .color))
        } catch {
            print("Failed to create heatmap texture: \(error)")
        }
    }
    
    private func createRiskDataBuffer() -> MTLBuffer? {
        guard let device = device else { return nil }
        
        let bufferSize = currentRiskData.count * MemoryLayout<MetalRiskDataPoint>.size
        let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        
        let bufferPointer = buffer?.contents().bindMemory(to: MetalRiskDataPoint.self, capacity: currentRiskData.count)
        
        for (index, dataPoint) in currentRiskData.enumerated() {
            bufferPointer?[index] = MetalRiskDataPoint(
                position: dataPoint.position,
                intensity: dataPoint.intensity,
                riskLevel: Int32(dataPoint.riskLevel.rawValue),
                confidence: dataPoint.confidence
            )
        }
        
        return buffer
    }
    
    private func createHeatmapPlane() {
        guard let heatmapTexture = heatmapTexture else { return }
        
        let mesh = MeshResource.generatePlane(width: 0.8, depth: 0.6)
        var material = SimpleMaterial()
        material.baseColor = MaterialColorParameter.texture(heatmapTexture)
        material.opacityThreshold = 0.1
        
        let heatmapModel = ModelEntity(mesh: mesh, materials: [material])
        heatmapModel.position = [0, 0, -0.1]
        
        heatmapEntity = heatmapModel
        documentAnchor?.addChild(heatmapModel)
    }
    
    private func createRiskIndicators() {
        for dataPoint in currentRiskData {
            let indicator = createRiskIndicator(riskLevel: dataPoint.riskLevel, at: dataPoint.position)
            documentAnchor?.addChild(indicator)
            riskIndicators[indicator.id] = indicator
        }
    }
    
    private func createRiskIndicator(riskLevel: RiskLevel, at position: SIMD3<Float>) -> RiskIndicatorEntity {
        let indicator = RiskIndicatorEntity(riskLevel: riskLevel)
        indicator.position = position
        
        // Create visual representation
        let radius = Float(riskLevel.rawValue) * 0.01 + 0.02
        let mesh = MeshResource.generateSphere(radius: radius)
        let material = SimpleMaterial(color: riskLevel.color.withAlphaComponent(0.8), isMetallic: false)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        indicator.addChild(modelEntity)
        
        // Add pulsing effect
        let pulseAnimation = createPulseAnimation(for: riskLevel)
        let controller = indicator.playAnimation(pulseAnimation)
        
        return indicator
    }
    
    private func createRiskAreaHighlight(center: SIMD3<Float>, radius: Float, riskLevel: RiskLevel) -> Entity {
        let entity = Entity()
        entity.position = center
        
        // Create expanding ring effect
        let ringMesh = MeshResource.generatePlane(width: radius * 2, depth: radius * 2, cornerRadius: radius)
        let ringMaterial = SimpleMaterial(color: riskLevel.color.withAlphaComponent(0.3), isMetallic: false)
        
        let ringEntity = ModelEntity(mesh: ringMesh, materials: [ringMaterial])
        entity.addChild(ringEntity)
        
        // Create expanding animation
        let expandAnimation = AnimationResource.makeTransform(
            duration: 2.0,
            repeatMode: .none,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(0.1, 0.1, 0.1))),
                .init(time: 2.0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        entity.playAnimation(expandAnimation)
        
        return entity
    }
    
    private func createHeatmapLegend() {
        let legendEntity = Entity()
        legendEntity.position = [0.5, 0, 0]
        
        let legendHeight: Float = 0.4
        let legendWidth: Float = 0.05
        
        // Create legend bar
        let legendMesh = MeshResource.generateBox(width: legendWidth, height: legendHeight, depth: 0.01)
        
        // Create gradient texture for legend
        let legendTexture = generateLegendTexture()
        var legendMaterial = SimpleMaterial()
        legendMaterial.baseColor = MaterialColorParameter.texture(legendTexture)
        
        let legendModel = ModelEntity(mesh: legendMesh, materials: [legendMaterial])
        legendEntity.addChild(legendModel)
        
        // Add labels
        createLegendLabels(legendEntity, height: legendHeight)
        
        documentAnchor?.addChild(legendEntity)
    }
    
    private func generateLegendTexture() -> TextureResource {
        // Create a vertical gradient texture for the legend
        let width = 32
        let height = 256
        
        var pixelData: [UInt8] = []
        
        for y in 0..<height {
            let t = Float(y) / Float(height - 1)
            let color = interpolateColor(t: t, scheme: getCurrentColorScheme())
            
            let r = UInt8(color.x * 255)
            let g = UInt8(color.y * 255)
            let b = UInt8(color.z * 255)
            let a = UInt8(color.w * 255)
            
            for _ in 0..<width {
                pixelData.append(r)
                pixelData.append(g)
                pixelData.append(b)
                pixelData.append(a)
            }
        }
        
        // Create texture from pixel data
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        
        guard let device = device,
              let texture = device.makeTexture(descriptor: textureDescriptor) else {
            fatalError("Failed to create legend texture")
        }
        
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), 
                              size: MTLSize(width: width, height: height, depth: 1))
        texture.replace(region: region, mipmapLevel: 0, withBytes: pixelData, bytesPerRow: width * 4)
        
        do {
            return try TextureResource.generate(from: texture, options: .init(semantic: .color))
        } catch {
            fatalError("Failed to create TextureResource: \(error)")
        }
    }
    
    private func createLegendLabels(_ legendEntity: Entity, height: Float) {
        let labels = ["Critical", "High", "Medium", "Low"]
        let positions: [Float] = [height/2 - 0.05, height/6, -height/6, -height/2 + 0.05]
        
        for (label, position) in zip(labels, positions) {
            let textMesh = MeshResource.generateText(
                label,
                extrusionDepth: 0.001,
                font: .systemFont(ofSize: 0.03),
                containerFrame: .zero,
                alignment: .center,
                lineBreakMode: .byTruncatingTail
            )
            
            let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
            let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
            textEntity.position = [0.08, position, 0]
            
            legendEntity.addChild(textEntity)
        }
    }
    
    private func getCurrentColorScheme() -> HeatmapColorScheme {
        switch heatmapMode {
        case .risk:
            return riskColorScheme
        case .confidence:
            return confidenceColorScheme
        case .combined:
            return riskColorScheme // Could blend both schemes
        }
    }
    
    private func interpolateColor(t: Float, scheme: HeatmapColorScheme) -> SIMD4<Float> {
        if t <= 0.25 {
            let localT = t / 0.25
            return mix(scheme.low, scheme.medium, t: localT)
        } else if t <= 0.5 {
            let localT = (t - 0.25) / 0.25
            return mix(scheme.medium, scheme.high, t: localT)
        } else {
            let localT = (t - 0.5) / 0.5
            return mix(scheme.high, scheme.critical, t: localT)
        }
    }
    
    private func createPulseAnimation(for riskLevel: RiskLevel) -> AnimationResource {
        let duration: TimeInterval
        let scaleRange: Float
        
        switch riskLevel {
        case .low:
            duration = 3.0
            scaleRange = 0.1
        case .medium:
            duration = 2.0
            scaleRange = 0.15
        case .high:
            duration = 1.5
            scaleRange = 0.2
        case .critical:
            duration = 1.0
            scaleRange = 0.3
        }
        
        return AnimationResource.makeTransform(
            duration: duration,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: duration/2, value: Transform(scale: SIMD3<Float>(1.0 + scaleRange, 1.0 + scaleRange, 1.0 + scaleRange))),
                .init(time: duration, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
    }
    
    private func animateIndicatorAppearance(_ indicator: RiskIndicatorEntity) {
        let appearAnimation = AnimationResource.makeTransform(
            duration: 0.5,
            repeatMode: .none,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(0, 0, 0))),
                .init(time: 0.3, value: Transform(scale: SIMD3<Float>(1.2, 1.2, 1.2))),
                .init(time: 0.5, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        indicator.playAnimation(appearAnimation)
    }
    
    private func animateIndicatorDisappearance(_ indicator: RiskIndicatorEntity, completion: @escaping () -> Void) {
        let disappearAnimation = AnimationResource.makeTransform(
            duration: 0.3,
            repeatMode: .none,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 0.3, value: Transform(scale: SIMD3<Float>(0, 0, 0)))
            ]
        )
        
        let controller = indicator.playAnimation(disappearAnimation)
        
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            completion()
        }
    }
    
    private func updateHeatmapVisualization() {
        regenerateHeatmapTexture()
        updateRiskIndicators()
    }
    
    private func updateHeatmapColors() {
        // Update all risk indicators with new intensity
        for (_, indicator) in riskIndicators {
            guard let modelEntity = indicator.children.first as? ModelEntity else { continue }
            
            let adjustedColor = indicator.riskLevel.color.withAlphaComponent(CGFloat(heatmapIntensity * 0.8))
            let material = SimpleMaterial(color: adjustedColor, isMetallic: false)
            modelEntity.model?.materials = [material]
        }
    }
    
    private func regenerateHeatmapTexture() {
        generateHeatmapTexture()
        
        if let heatmapModel = heatmapEntity as? ModelEntity,
           let heatmapTexture = heatmapTexture {
            var material = SimpleMaterial()
            material.baseColor = MaterialColorParameter.texture(heatmapTexture)
            heatmapModel.model?.materials = [material]
        }
    }
    
    private func updateRiskIndicators() {
        // Remove existing indicators
        for (_, indicator) in riskIndicators {
            indicator.removeFromParent()
        }
        riskIndicators.removeAll()
        
        // Create new indicators based on current data
        createRiskIndicators()
    }
    
    private func startHeatmapAnimations() {
        // Start global heatmap animation
        guard let heatmapModel = heatmapEntity as? ModelEntity else { return }
        
        let breathingAnimation = AnimationResource.makeTransform(
            duration: 4.0 / animationSpeed,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: true,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 2.0 / animationSpeed, value: Transform(scale: SIMD3<Float>(1.02, 1.02, 1.0))),
                .init(time: 4.0 / animationSpeed, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        animationController = heatmapModel.playAnimation(breathingAnimation)
    }
    
    private func stopAnimations() {
        animationController?.stop()
        animationController = nil
    }
    
    private func clearHeatmapVisualization() {
        heatmapEntity?.removeFromParent()
        heatmapEntity = nil
        
        for (_, indicator) in riskIndicators {
            indicator.removeFromParent()
        }
        riskIndicators.removeAll()
        
        documentAnchor?.removeFromParent()
        documentAnchor = nil
    }
}

// MARK: - Supporting Types

public enum HeatmapMode: Int, CaseIterable {
    case risk = 0
    case confidence = 1
    case combined = 2
    
    public var displayName: String {
        switch self {
        case .risk: return "Risk Level"
        case .confidence: return "Confidence"
        case .combined: return "Combined"
        }
    }
}

public struct RiskDataPoint: Identifiable {
    public let id: UUID
    public let position: SIMD3<Float>
    public let riskLevel: RiskLevel
    public let confidence: Float
    public let intensity: Float
    public let metadata: RiskMetadata
    
    public init(id: UUID = UUID(), position: SIMD3<Float>, riskLevel: RiskLevel, confidence: Float, intensity: Float, metadata: RiskMetadata) {
        self.id = id
        self.position = position
        self.riskLevel = riskLevel
        self.confidence = confidence
        self.intensity = intensity
        self.metadata = metadata
    }
}

public struct RiskMetadata {
    public let clauseTitle: String
    public let section: String
    public let content: String
    
    public init(clauseTitle: String, section: String, content: String) {
        self.clauseTitle = clauseTitle
        self.section = section
        self.content = content
    }
}

private struct HeatmapColorScheme {
    let low: SIMD4<Float>
    let medium: SIMD4<Float>
    let high: SIMD4<Float>
    let critical: SIMD4<Float>
}

private struct HeatmapParameters {
    let intensity: Float
    let threshold: Float
    let mode: Int32
    let dataCount: Int32
}

private struct MetalRiskDataPoint {
    let position: SIMD3<Float>
    let intensity: Float
    let riskLevel: Int32
    let confidence: Float
}

private class RiskIndicatorEntity: Entity, HasModel {
    let id = UUID()
    let riskLevel: RiskLevel
    
    init(riskLevel: RiskLevel) {
        self.riskLevel = riskLevel
        super.init()
    }
    
    required init() {
        fatalError("init() has not been implemented")
    }
}

// MARK: - Metal Shader Helpers

private func mix<T: SIMD>(_ a: T, _ b: T, t: Float) -> T where T.Scalar == Float {
    return a + (b - a) * t
}