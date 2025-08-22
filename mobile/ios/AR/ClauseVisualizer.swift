//
//  ClauseVisualizer.swift
//  AR Document Analysis Framework
//
//  Advanced 3D clause visualization and relationship graphs in AR space
//

import ARKit
import RealityKit
import SwiftUI
import Combine

@available(iOS 13.0, *)
public class ClauseVisualizer: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isVisualizationActive = false
    @Published public var currentDocument: AnalyzedDocument?
    @Published public var visualizationMode: VisualizationMode = .hierarchical
    @Published public var selectedClause: ClauseNode?
    @Published public var relationshipStrength: Float = 1.0
    
    private var arView: ARView?
    private var documentAnchor: AnchorEntity?
    private var clauseNodes: [UUID: ClauseNodeEntity] = [:]
    private var relationshipLines: [UUID: RelationshipEntity] = [:]
    private var animationControllers: [AnimationPlaybackController] = []
    
    // Spatial configuration
    private let baseRadius: Float = 0.5
    private let heightScale: Float = 0.3
    private let nodeSize: Float = 0.05
    private let connectionWidth: Float = 0.005
    
    // Interaction
    private var gestureRecognizer: UITapGestureRecognizer?
    private var panGestureRecognizer: UIPanGestureRecognizer?
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
    }
    
    // MARK: - Public Methods
    
    public func startVisualization(for document: AnalyzedDocument, in arView: ARView) {
        self.arView = arView
        self.currentDocument = document
        
        setupGestureRecognizers(in: arView)
        createDocumentVisualization(document)
        
        isVisualizationActive = true
    }
    
    public func stopVisualization() {
        clearVisualization()
        isVisualizationActive = false
        gestureRecognizer = nil
        panGestureRecognizer = nil
    }
    
    public func switchVisualizationMode(_ mode: VisualizationMode) {
        visualizationMode = mode
        
        guard let document = currentDocument else { return }
        
        clearVisualization()
        createDocumentVisualization(document)
    }
    
    public func highlightClause(_ clauseId: UUID, animated: Bool = true) {
        guard let nodeEntity = clauseNodes[clauseId] else { return }
        
        selectedClause = currentDocument?.clauses.first { $0.id == clauseId }
        
        // Reset all nodes to default state
        resetAllHighlights()
        
        // Highlight selected node
        highlightNode(nodeEntity, color: .systemYellow, animated: animated)
        
        // Highlight related clauses
        highlightRelatedClauses(for: clauseId, animated: animated)
    }
    
    public func showClauseRelationships(for clauseId: UUID, strength: Float = 0.5) {
        relationshipStrength = strength
        
        guard let document = currentDocument,
              let clause = document.clauses.first(where: { $0.id == clauseId }) else { return }
        
        // Hide all relationships first
        hideAllRelationships()
        
        // Show relationships above threshold
        for relationship in clause.relationships where relationship.strength >= strength {
            showRelationship(relationship, animated: true)
        }
    }
    
    public func animateDocumentFlow() {
        guard let document = currentDocument else { return }
        
        // Create flow animation through document structure
        animateClauseSequence(document.clauses.sorted { $0.order < $1.order })
    }
    
    public func create3DDocumentStructure() -> Entity? {
        guard let document = currentDocument else { return nil }
        
        let containerEntity = Entity()
        
        switch visualizationMode {
        case .hierarchical:
            create3DHierarchy(document, in: containerEntity)
        case .network:
            create3DNetwork(document, in: containerEntity)
        case .timeline:
            create3DTimeline(document, in: containerEntity)
        case .riskBased:
            create3DRiskVisualization(document, in: containerEntity)
        case .comparative:
            create3DComparative(document, in: containerEntity)
        }
        
        return containerEntity
    }
    
    // MARK: - Private Methods
    
    private func createDocumentVisualization(_ document: AnalyzedDocument) {
        guard let arView = arView else { return }
        
        // Create anchor at user's position
        let cameraTransform = arView.session.currentFrame?.camera.transform ?? matrix_identity_float4x4
        let anchorEntity = AnchorEntity(world: cameraTransform)
        
        // Position anchor in front of user
        anchorEntity.position = [0, 0, -1.5]
        
        // Create document structure
        if let documentStructure = create3DDocumentStructure() {
            anchorEntity.addChild(documentStructure)
        }
        
        // Add floating title
        let titleEntity = createDocumentTitle(document.title)
        titleEntity.position = [0, 0.8, 0]
        anchorEntity.addChild(titleEntity)
        
        arView.scene.addAnchor(anchorEntity)
        self.documentAnchor = anchorEntity
        
        // Start idle animations
        startIdleAnimations()
    }
    
    private func create3DHierarchy(_ document: AnalyzedDocument, in container: Entity) {
        let sections = Dictionary(grouping: document.clauses) { $0.section }
        let sectionCount = sections.count
        
        for (index, (sectionName, clauses)) in sections.enumerated() {
            let angleStep = 2 * Float.pi / Float(sectionCount)
            let angle = angleStep * Float(index)
            
            let sectionX = cos(angle) * baseRadius
            let sectionZ = sin(angle) * baseRadius
            
            // Create section header
            let sectionEntity = createSectionNode(sectionName, position: [sectionX, 0.4, sectionZ])
            container.addChild(sectionEntity)
            
            // Create clauses in vertical stack
            for (clauseIndex, clause) in clauses.enumerated() {
                let clauseY = -Float(clauseIndex) * 0.15
                let clausePosition: SIMD3<Float> = [sectionX, clauseY, sectionZ]
                
                let clauseEntity = createClauseNode(clause, at: clausePosition)
                container.addChild(clauseEntity)
                
                clauseNodes[clause.id] = clauseEntity
                
                // Create connections to section
                if clauseIndex == 0 {
                    let connection = createConnection(
                        from: [sectionX, 0.4, sectionZ],
                        to: clausePosition,
                        type: .hierarchical
                    )
                    container.addChild(connection)
                }
            }
        }
    }
    
    private func create3DNetwork(_ document: AnalyzedDocument, in container: Entity) {
        // Position clauses in 3D space based on relationships
        let positions = calculateNetworkPositions(document.clauses)
        
        for (clause, position) in zip(document.clauses, positions) {
            let clauseEntity = createClauseNode(clause, at: position)
            container.addChild(clauseEntity)
            clauseNodes[clause.id] = clauseEntity
        }
        
        // Create relationship connections
        for clause in document.clauses {
            for relationship in clause.relationships {
                guard let targetPosition = positions.first(where: { 
                    document.clauses[positions.firstIndex(of: $0)!].id == relationship.targetClauseId 
                }) else { continue }
                
                let sourcePosition = positions[document.clauses.firstIndex { $0.id == clause.id }!]
                
                let connection = createConnection(
                    from: sourcePosition,
                    to: targetPosition,
                    type: .relationship(relationship.strength)
                )
                container.addChild(connection)
                
                if let relationshipEntity = connection as? RelationshipEntity {
                    relationshipLines[relationship.id] = relationshipEntity
                }
            }
        }
    }
    
    private func create3DTimeline(_ document: AnalyzedDocument, in container: Entity) {
        let sortedClauses = document.clauses.sorted { $0.order < $1.order }
        
        for (index, clause) in sortedClauses.enumerated() {
            let progress = Float(index) / Float(sortedClauses.count - 1)
            let angle = progress * 2 * Float.pi
            
            let x = cos(angle) * baseRadius
            let y = sin(progress * Float.pi) * heightScale
            let z = sin(angle) * baseRadius
            
            let position: SIMD3<Float> = [x, y, z]
            let clauseEntity = createClauseNode(clause, at: position)
            container.addChild(clauseEntity)
            clauseNodes[clause.id] = clauseEntity
            
            // Create timeline path
            if index < sortedClauses.count - 1 {
                let nextIndex = index + 1
                let nextProgress = Float(nextIndex) / Float(sortedClauses.count - 1)
                let nextAngle = nextProgress * 2 * Float.pi
                
                let nextX = cos(nextAngle) * baseRadius
                let nextY = sin(nextProgress * Float.pi) * heightScale
                let nextZ = sin(nextAngle) * baseRadius
                
                let nextPosition: SIMD3<Float> = [nextX, nextY, nextZ]
                
                let connection = createConnection(
                    from: position,
                    to: nextPosition,
                    type: .timeline
                )
                container.addChild(connection)
            }
        }
    }
    
    private func create3DRiskVisualization(_ document: AnalyzedDocument, in container: Entity) {
        let riskLevels = Dictionary(grouping: document.clauses) { $0.riskLevel }
        
        for (risk, clauses) in riskLevels {
            let riskHeight = Float(risk.rawValue) * 0.2
            let clauseCount = clauses.count
            
            for (index, clause) in clauses.enumerated() {
                let angleStep = 2 * Float.pi / Float(clauseCount)
                let angle = angleStep * Float(index)
                let radius = baseRadius * (1.0 - Float(risk.rawValue) * 0.1)
                
                let x = cos(angle) * radius
                let z = sin(angle) * radius
                
                let position: SIMD3<Float> = [x, riskHeight, z]
                let clauseEntity = createClauseNode(clause, at: position)
                container.addChild(clauseEntity)
                clauseNodes[clause.id] = clauseEntity
            }
            
            // Create risk level indicator
            let riskIndicator = createRiskLevelIndicator(risk, at: [0, riskHeight, 0])
            container.addChild(riskIndicator)
        }
    }
    
    private func create3DComparative(_ document: AnalyzedDocument, in container: Entity) {
        // Create side-by-side comparison of different document versions or sections
        let sections = Array(Set(document.clauses.map { $0.section }))
        
        for (sectionIndex, section) in sections.enumerated() {
            let sectionClauses = document.clauses.filter { $0.section == section }
            let offsetX = Float(sectionIndex - sections.count / 2) * 0.8
            
            for (clauseIndex, clause) in sectionClauses.enumerated() {
                let position: SIMD3<Float> = [
                    offsetX,
                    -Float(clauseIndex) * 0.1,
                    0
                ]
                
                let clauseEntity = createClauseNode(clause, at: position)
                container.addChild(clauseEntity)
                clauseNodes[clause.id] = clauseEntity
            }
        }
    }
    
    private func createClauseNode(_ clause: ClauseData, at position: SIMD3<Float>) -> ClauseNodeEntity {
        let nodeEntity = ClauseNodeEntity(clause: clause)
        nodeEntity.position = position
        
        // Create visual representation based on clause properties
        let mesh = MeshResource.generateSphere(radius: nodeSize)
        let material = createClauseMaterial(for: clause)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        nodeEntity.addChild(modelEntity)
        
        // Add text label
        let textMesh = MeshResource.generateText(
            String(clause.title.prefix(20)),
            extrusionDepth: 0.001,
            font: .systemFont(ofSize: 0.05),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        textEntity.position = [0, nodeSize + 0.02, 0]
        
        nodeEntity.addChild(textEntity)
        
        // Add interaction component
        modelEntity.components.set(InputTargetComponent())
        
        return nodeEntity
    }
    
    private func createSectionNode(_ sectionName: String, position: SIMD3<Float>) -> Entity {
        let entity = Entity()
        entity.position = position
        
        // Create larger node for section
        let mesh = MeshResource.generateSphere(radius: nodeSize * 1.5)
        let material = SimpleMaterial(color: .systemBlue, isMetallic: true)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        entity.addChild(modelEntity)
        
        // Add section label
        let textMesh = MeshResource.generateText(
            sectionName,
            extrusionDepth: 0.002,
            font: .boldSystemFont(ofSize: 0.06),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        textEntity.position = [0, nodeSize * 1.5 + 0.03, 0]
        
        entity.addChild(textEntity)
        
        return entity
    }
    
    private func createConnection(from startPos: SIMD3<Float>, to endPos: SIMD3<Float>, type: ConnectionType) -> Entity {
        let connectionEntity = RelationshipEntity(from: startPos, to: endPos, type: type)
        
        let distance = distance(startPos, endPos)
        let direction = normalize(endPos - startPos)
        let midpoint = (startPos + endPos) / 2
        
        // Create cylinder for connection
        let mesh = MeshResource.generateCylinder(height: distance, radius: connectionWidth)
        let material = createConnectionMaterial(for: type)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        
        // Orient cylinder between points
        let up = SIMD3<Float>(0, 1, 0)
        let rotationAxis = cross(up, direction)
        let rotationAngle = acos(dot(up, direction))
        
        if length(rotationAxis) > 0.001 {
            let normalizedAxis = normalize(rotationAxis)
            let quaternion = simd_quatf(angle: rotationAngle, axis: normalizedAxis)
            modelEntity.orientation = quaternion
        }
        
        modelEntity.position = midpoint
        connectionEntity.addChild(modelEntity)
        
        return connectionEntity
    }
    
    private func createClauseMaterial(for clause: ClauseData) -> Material {
        var color: UIColor
        
        switch clause.riskLevel {
        case .low:
            color = .systemGreen
        case .medium:
            color = .systemOrange
        case .high:
            color = .systemRed
        case .critical:
            color = .systemPurple
        }
        
        // Add transparency based on confidence
        color = color.withAlphaComponent(CGFloat(clause.confidence))
        
        return SimpleMaterial(color: color, isMetallic: false)
    }
    
    private func createConnectionMaterial(for type: ConnectionType) -> Material {
        let color: UIColor
        let metallic: Bool
        
        switch type {
        case .hierarchical:
            color = .systemBlue
            metallic = false
        case .relationship(let strength):
            let alpha = CGFloat(strength)
            color = UIColor.systemPurple.withAlphaComponent(alpha)
            metallic = true
        case .timeline:
            color = .systemTeal
            metallic = false
        }
        
        return SimpleMaterial(color: color, isMetallic: metallic)
    }
    
    private func createDocumentTitle(_ title: String) -> Entity {
        let entity = Entity()
        
        let textMesh = MeshResource.generateText(
            title,
            extrusionDepth: 0.005,
            font: .boldSystemFont(ofSize: 0.1),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: true)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        
        entity.addChild(textEntity)
        
        return entity
    }
    
    private func createRiskLevelIndicator(_ riskLevel: RiskLevel, at position: SIMD3<Float>) -> Entity {
        let entity = Entity()
        entity.position = position
        
        let mesh = MeshResource.generateBox(width: 0.1, height: 0.02, depth: 0.1)
        let material = SimpleMaterial(color: riskLevel.color, isMetallic: false)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        entity.addChild(modelEntity)
        
        return entity
    }
    
    private func calculateNetworkPositions(_ clauses: [ClauseData]) -> [SIMD3<Float>] {
        // Implement force-directed graph layout algorithm
        var positions: [SIMD3<Float>] = []
        
        // Initialize random positions
        for _ in clauses {
            let randomX = Float.random(in: -baseRadius...baseRadius)
            let randomY = Float.random(in: -heightScale...heightScale)
            let randomZ = Float.random(in: -baseRadius...baseRadius)
            positions.append([randomX, randomY, randomZ])
        }
        
        // Apply force-directed algorithm iterations
        for _ in 0..<100 {
            var forces: [SIMD3<Float>] = Array(repeating: [0, 0, 0], count: clauses.count)
            
            // Repulsion forces
            for i in 0..<positions.count {
                for j in (i+1)..<positions.count {
                    let delta = positions[i] - positions[j]
                    let distance = length(delta)
                    if distance > 0.001 {
                        let force = normalize(delta) * (0.1 / (distance * distance))
                        forces[i] += force
                        forces[j] -= force
                    }
                }
            }
            
            // Attraction forces based on relationships
            for (i, clause) in clauses.enumerated() {
                for relationship in clause.relationships {
                    if let j = clauses.firstIndex(where: { $0.id == relationship.targetClauseId }) {
                        let delta = positions[j] - positions[i]
                        let distance = length(delta)
                        if distance > 0.001 {
                            let force = normalize(delta) * relationship.strength * 0.05
                            forces[i] += force
                            forces[j] -= force
                        }
                    }
                }
            }
            
            // Apply forces
            for i in 0..<positions.count {
                positions[i] += forces[i] * 0.1
            }
        }
        
        return positions
    }
    
    private func setupGestureRecognizers(in arView: ARView) {
        gestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        panGestureRecognizer = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        
        arView.addGestureRecognizer(gestureRecognizer!)
        arView.addGestureRecognizer(panGestureRecognizer!)
    }
    
    private func clearVisualization() {
        documentAnchor?.removeFromParent()
        documentAnchor = nil
        clauseNodes.removeAll()
        relationshipLines.removeAll()
        
        // Stop all animations
        animationControllers.forEach { $0.stop() }
        animationControllers.removeAll()
    }
    
    private func resetAllHighlights() {
        for (_, nodeEntity) in clauseNodes {
            resetNodeHighlight(nodeEntity)
        }
    }
    
    private func highlightNode(_ node: ClauseNodeEntity, color: UIColor, animated: Bool) {
        guard let modelEntity = node.children.first as? ModelEntity else { return }
        
        let highlightMaterial = SimpleMaterial(color: color, isMetallic: true)
        modelEntity.model?.materials = [highlightMaterial]
        
        if animated {
            let scaleUp = Transform(scale: SIMD3<Float>(1.2, 1.2, 1.2), translation: node.position)
            let scaleDown = Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0), translation: node.position)
            
            let animation = AnimationResource.makeTransform(
                duration: 0.3,
                repeatMode: .none,
                blendLayer: 0,
                additive: false,
                bindTarget: .transform,
                keyframes: [
                    .init(time: 0, value: scaleDown),
                    .init(time: 0.15, value: scaleUp),
                    .init(time: 0.3, value: scaleDown)
                ]
            )
            
            let controller = node.playAnimation(animation)
            animationControllers.append(controller)
        }
    }
    
    private func resetNodeHighlight(_ node: ClauseNodeEntity) {
        guard let modelEntity = node.children.first as? ModelEntity else { return }
        
        let originalMaterial = createClauseMaterial(for: node.clause)
        modelEntity.model?.materials = [originalMaterial]
    }
    
    private func highlightRelatedClauses(for clauseId: UUID, animated: Bool) {
        guard let document = currentDocument,
              let clause = document.clauses.first(where: { $0.id == clauseId }) else { return }
        
        for relationship in clause.relationships {
            if let relatedNode = clauseNodes[relationship.targetClauseId] {
                let intensity = relationship.strength
                let color = UIColor.systemBlue.withAlphaComponent(CGFloat(intensity))
                highlightNode(relatedNode, color: color, animated: animated)
            }
        }
    }
    
    private func hideAllRelationships() {
        for (_, relationship) in relationshipLines {
            relationship.isEnabled = false
        }
    }
    
    private func showRelationship(_ relationship: ClauseRelationship, animated: Bool) {
        guard let relationshipEntity = relationshipLines[relationship.id] else { return }
        
        relationshipEntity.isEnabled = true
        
        if animated {
            let fadeInAnimation = AnimationResource.makeOpacity(
                duration: 0.5,
                repeatMode: .none,
                blendLayer: 0,
                additive: false,
                bindTarget: .opacity,
                keyframes: [
                    .init(time: 0, value: 0),
                    .init(time: 0.5, value: relationship.strength)
                ]
            )
            
            let controller = relationshipEntity.playAnimation(fadeInAnimation)
            animationControllers.append(controller)
        }
    }
    
    private func animateClauseSequence(_ clauses: [ClauseData]) {
        for (index, clause) in clauses.enumerated() {
            guard let nodeEntity = clauseNodes[clause.id] else { continue }
            
            let delay = TimeInterval(index) * 0.2
            
            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                self.highlightNode(nodeEntity, color: .systemYellow, animated: true)
                
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                    self.resetNodeHighlight(nodeEntity)
                }
            }
        }
    }
    
    private func startIdleAnimations() {
        // Add subtle floating animation to all nodes
        for (_, nodeEntity) in clauseNodes {
            let floatAnimation = createFloatingAnimation()
            let controller = nodeEntity.playAnimation(floatAnimation)
            animationControllers.append(controller)
        }
    }
    
    private func createFloatingAnimation() -> AnimationResource {
        return AnimationResource.makeTransform(
            duration: 3.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: true,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(translation: [0, 0, 0])),
                .init(time: 1.5, value: Transform(translation: [0, 0.02, 0])),
                .init(time: 3.0, value: Transform(translation: [0, 0, 0]))
            ]
        )
    }
    
    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let arView = arView else { return }
        
        let location = gesture.location(in: arView)
        
        if let entity = arView.entity(at: location) as? ClauseNodeEntity {
            highlightClause(entity.clause.id, animated: true)
        }
    }
    
    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        // Implement rotation of the entire visualization
        guard let arView = arView, let anchor = documentAnchor else { return }
        
        let translation = gesture.translation(in: arView)
        let velocity = gesture.velocity(in: arView)
        
        switch gesture.state {
        case .changed:
            let rotationX = Float(translation.y) * 0.01
            let rotationY = Float(translation.x) * 0.01
            
            let rotationQuaternion = simd_quatf(angle: rotationY, axis: [0, 1, 0]) * simd_quatf(angle: rotationX, axis: [1, 0, 0])
            anchor.orientation = rotationQuaternion * anchor.orientation
            
        case .ended:
            // Add momentum-based rotation
            let velocityX = Float(velocity.y) * 0.001
            let velocityY = Float(velocity.x) * 0.001
            
            let momentumRotation = simd_quatf(angle: velocityY, axis: [0, 1, 0]) * simd_quatf(angle: velocityX, axis: [1, 0, 0])
            
            let momentumAnimation = AnimationResource.makeTransform(
                duration: 1.0,
                repeatMode: .none,
                blendLayer: 0,
                additive: true,
                bindTarget: .transform,
                keyframes: [
                    .init(time: 0, value: Transform(rotation: anchor.orientation)),
                    .init(time: 1.0, value: Transform(rotation: momentumRotation * anchor.orientation))
                ]
            )
            
            let controller = anchor.playAnimation(momentumAnimation)
            animationControllers.append(controller)
            
        default:
            break
        }
        
        gesture.setTranslation(.zero, in: arView)
    }
}

// MARK: - Supporting Types

public enum VisualizationMode: CaseIterable {
    case hierarchical
    case network
    case timeline
    case riskBased
    case comparative
    
    public var displayName: String {
        switch self {
        case .hierarchical: return "Hierarchical"
        case .network: return "Network"
        case .timeline: return "Timeline"
        case .riskBased: return "Risk Based"
        case .comparative: return "Comparative"
        }
    }
}

private enum ConnectionType {
    case hierarchical
    case relationship(Float)
    case timeline
}

public struct AnalyzedDocument {
    public let id: UUID
    public let title: String
    public let clauses: [ClauseData]
    public let createdAt: Date
    
    public init(id: UUID = UUID(), title: String, clauses: [ClauseData], createdAt: Date = Date()) {
        self.id = id
        self.title = title
        self.clauses = clauses
        self.createdAt = createdAt
    }
}

public struct ClauseData: Identifiable {
    public let id: UUID
    public let title: String
    public let content: String
    public let section: String
    public let order: Int
    public let riskLevel: RiskLevel
    public let confidence: Float
    public let relationships: [ClauseRelationship]
    
    public init(id: UUID = UUID(), title: String, content: String, section: String, order: Int, riskLevel: RiskLevel, confidence: Float, relationships: [ClauseRelationship] = []) {
        self.id = id
        self.title = title
        self.content = content
        self.section = section
        self.order = order
        self.riskLevel = riskLevel
        self.confidence = confidence
        self.relationships = relationships
    }
}

public struct ClauseRelationship: Identifiable {
    public let id: UUID
    public let targetClauseId: UUID
    public let type: RelationshipType
    public let strength: Float
    public let description: String
    
    public init(id: UUID = UUID(), targetClauseId: UUID, type: RelationshipType, strength: Float, description: String) {
        self.id = id
        self.targetClauseId = targetClauseId
        self.type = type
        self.strength = strength
        self.description = description
    }
}

public enum RelationshipType {
    case references
    case conflicts
    case supports
    case modifies
    case depends
}

public enum RiskLevel: Int, CaseIterable {
    case low = 1
    case medium = 2
    case high = 3
    case critical = 4
    
    public var color: UIColor {
        switch self {
        case .low: return .systemGreen
        case .medium: return .systemOrange
        case .high: return .systemRed
        case .critical: return .systemPurple
        }
    }
    
    public var displayName: String {
        switch self {
        case .low: return "Low Risk"
        case .medium: return "Medium Risk"
        case .high: return "High Risk"
        case .critical: return "Critical Risk"
        }
    }
}

// MARK: - Custom Entity Classes

private class ClauseNodeEntity: Entity, HasModel {
    let clause: ClauseData
    
    init(clause: ClauseData) {
        self.clause = clause
        super.init()
    }
    
    required init() {
        fatalError("init() has not been implemented")
    }
}

private class RelationshipEntity: Entity, HasModel {
    let connectionType: ConnectionType
    let startPosition: SIMD3<Float>
    let endPosition: SIMD3<Float>
    
    init(from startPos: SIMD3<Float>, to endPos: SIMD3<Float>, type: ConnectionType) {
        self.startPosition = startPos
        self.endPosition = endPos
        self.connectionType = type
        super.init()
    }
    
    required init() {
        fatalError("init() has not been implemented")
    }
}