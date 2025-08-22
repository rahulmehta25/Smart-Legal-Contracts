//
//  Collaboration.swift
//  AR Document Analysis Framework
//
//  Shared AR sessions and multi-user document collaboration
//

import ARKit
import RealityKit
import MultipeerConnectivity
import SwiftUI
import Combine
import AVFoundation

@available(iOS 13.0, *)
public class ARCollaborationManager: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isSessionActive = false
    @Published public var connectedPeers: [CollaborationPeer] = []
    @Published public var sessionMode: CollaborationMode = .viewer
    @Published public var spatialAudio = true
    @Published public var screenSharing = false
    @Published public var remoteAssistanceActive = false
    
    private var arView: ARView?
    private var mcSession: MCSession?
    private var mcAdvertiser: MCNearbyServiceAdvertiser?
    private var mcBrowser: MCNearbyServiceBrowser?
    private var peerID: MCPeerID
    
    // ARKit Collaborative Session
    private var collaborationData: ARSession.CollaborationData?
    private var collaborativeSession: ARSession?
    
    // Shared AR Anchors
    private var sharedAnchors: [UUID: SharedAnchorEntity] = [:]
    private var remoteUserAnchors: [MCPeerID: UserAnchorEntity] = [:]
    
    // Annotations and Comments
    private var sharedAnnotations: [UUID: ARAnnotation] = [:]
    private var activeComments: [UUID: ARComment] = [:]
    
    // Voice and Video
    private var audioEngine: AVAudioEngine?
    private var spatialAudioMixer: AVAudioEnvironmentNode?
    private var voiceRecorder: AVAudioRecorder?
    private var voicePlayer: [MCPeerID: AVAudioPlayer] = [:]
    
    // Screen Sharing
    private var screenRecorder: ScreenRecorder?
    private var remoteScreens: [MCPeerID: Entity] = [:]
    
    // Data synchronization
    private let syncQueue = DispatchQueue(label: "com.documentanalyzer.collaboration.sync", qos: .userInteractive)
    private var lastSyncTimestamp: Date = Date()
    
    // Service type for Multipeer Connectivity
    private let serviceType = "doc-analyzer-ar"
    
    // MARK: - Initialization
    
    public override init() {
        let deviceName = UIDevice.current.name
        peerID = MCPeerID(displayName: deviceName)
        
        super.init()
        
        setupMultipeerConnectivity()
        setupSpatialAudio()
    }
    
    deinit {
        stopSession()
    }
    
    // MARK: - Public Methods
    
    public func startCollaborationSession(in arView: ARView, mode: CollaborationMode = .host) {
        self.arView = arView
        self.sessionMode = mode
        
        setupCollaborativeARSession()
        
        switch mode {
        case .host:
            startHosting()
        case .viewer:
            startBrowsing()
        }
        
        isSessionActive = true
        startSpatialAudio()
    }
    
    public func stopSession() {
        mcAdvertiser?.stopAdvertisingPeer()
        mcBrowser?.stopBrowsingForPeers()
        mcSession?.disconnect()
        
        stopSpatialAudio()
        clearSharedContent()
        
        isSessionActive = false
        connectedPeers.removeAll()
    }
    
    public func addAnnotation(at worldPosition: SIMD3<Float>, text: String, type: ARAnnotationType) -> UUID {
        let annotation = ARAnnotation(
            id: UUID(),
            position: worldPosition,
            text: text,
            type: type,
            authorId: peerID,
            timestamp: Date()
        )
        
        sharedAnnotations[annotation.id] = annotation
        createAnnotationEntity(annotation)
        
        // Broadcast to peers
        broadcastAnnotation(annotation)
        
        return annotation.id
    }
    
    public func addComment(to annotationId: UUID, text: String) {
        guard let annotation = sharedAnnotations[annotationId] else { return }
        
        let comment = ARComment(
            id: UUID(),
            annotationId: annotationId,
            text: text,
            authorId: peerID,
            timestamp: Date()
        )
        
        activeComments[comment.id] = comment
        updateAnnotationWithComment(annotation, comment: comment)
        
        // Broadcast to peers
        broadcastComment(comment)
    }
    
    public func shareDocument(_ document: AnalyzedDocument) {
        let sharedDoc = SharedDocument(
            id: document.id,
            title: document.title,
            clauses: document.clauses,
            sharedBy: peerID,
            timestamp: Date()
        )
        
        createSharedDocumentVisualization(sharedDoc)
        broadcastSharedDocument(sharedDoc)
    }
    
    public func enableScreenSharing() {
        guard !screenSharing else { return }
        
        screenRecorder = ScreenRecorder()
        screenRecorder?.delegate = self
        screenRecorder?.startRecording()
        
        screenSharing = true
        notifyPeersScreenSharingStarted()
    }
    
    public func disableScreenSharing() {
        screenRecorder?.stopRecording()
        screenRecorder = nil
        
        screenSharing = false
        notifyPeersScreenSharingStopped()
    }
    
    public func requestRemoteAssistance(from peer: MCPeerID) {
        let request = RemoteAssistanceRequest(
            requesterId: peerID,
            targetId: peer,
            timestamp: Date()
        )
        
        sendRemoteAssistanceRequest(request, to: peer)
    }
    
    public func acceptRemoteAssistance(from peer: MCPeerID) {
        remoteAssistanceActive = true
        
        // Enable enhanced collaboration features
        enableRemotePointing()
        enableVoiceGuidance()
        
        let response = RemoteAssistanceResponse(
            accepted: true,
            assistantId: peerID,
            requesterId: peer,
            timestamp: Date()
        )
        
        sendRemoteAssistanceResponse(response, to: peer)
    }
    
    public func pointAtLocation(_ worldPosition: SIMD3<Float>, duration: TimeInterval = 3.0) {
        let pointer = ARPointer(
            id: UUID(),
            position: worldPosition,
            authorId: peerID,
            timestamp: Date(),
            duration: duration
        )
        
        createPointerEntity(pointer)
        broadcastPointer(pointer)
    }
    
    public func recordVoiceMessage() {
        startVoiceRecording()
    }
    
    public func stopVoiceRecording() {
        finishVoiceRecording()
    }
    
    // MARK: - Private Methods
    
    private func setupMultipeerConnectivity() {
        mcSession = MCSession(peer: peerID, securityIdentity: nil, encryptionPreference: .required)
        mcSession?.delegate = self
        
        mcAdvertiser = MCNearbyServiceAdvertiser(peer: peerID, discoveryInfo: nil, serviceType: serviceType)
        mcAdvertiser?.delegate = self
        
        mcBrowser = MCNearbyServiceBrowser(peer: peerID, serviceType: serviceType)
        mcBrowser?.delegate = self
    }
    
    private func setupSpatialAudio() {
        audioEngine = AVAudioEngine()
        spatialAudioMixer = AVAudioEnvironmentNode()
        
        guard let audioEngine = audioEngine,
              let mixer = spatialAudioMixer else { return }
        
        audioEngine.attach(mixer)
        audioEngine.connect(mixer, to: audioEngine.outputNode, format: nil)
        
        // Configure 3D audio environment
        mixer.outputType = .headphones
        mixer.distanceAttenuationParameters.distanceAttenuationModel = .inverse
        mixer.distanceAttenuationParameters.maximumDistance = 10.0
        mixer.reverbParameters.enable = true
        mixer.reverbParameters.level = 20
    }
    
    private func setupCollaborativeARSession() {
        guard let arView = arView else { return }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.environmentTexturing = .automatic
        configuration.isCollaborationEnabled = true
        
        if ARWorldTrackingConfiguration.supportsUserFaceTracking {
            configuration.userFaceTrackingEnabled = true
        }
        
        collaborativeSession = arView.session
        arView.session.run(configuration, options: [.removeExistingAnchors, .resetTracking])
        
        // Set up collaboration data delegate
        arView.session.delegate = self
    }
    
    private func startHosting() {
        mcAdvertiser?.startAdvertisingPeer()
        print("Started advertising collaboration session")
    }
    
    private func startBrowsing() {
        mcBrowser?.startBrowsingForPeers()
        print("Started browsing for collaboration sessions")
    }
    
    private func createAnnotationEntity(_ annotation: ARAnnotation) {
        guard let arView = arView else { return }
        
        let anchorEntity = SharedAnchorEntity(annotation: annotation)
        anchorEntity.position = annotation.position
        
        // Create visual representation based on type
        let visualEntity = createAnnotationVisual(annotation)
        anchorEntity.addChild(visualEntity)
        
        // Create text label
        let textEntity = createAnnotationText(annotation)
        textEntity.position = [0, 0.05, 0]
        anchorEntity.addChild(textEntity)
        
        // Add interaction component
        visualEntity.components.set(InputTargetComponent())
        
        arView.scene.addAnchor(anchorEntity)
        sharedAnchors[annotation.id] = anchorEntity
    }
    
    private func createAnnotationVisual(_ annotation: ARAnnotation) -> Entity {
        let entity = Entity()
        
        let color: UIColor
        let size: Float
        
        switch annotation.type {
        case .highlight:
            color = .systemYellow
            size = 0.02
        case .warning:
            color = .systemOrange
            size = 0.025
        case .error:
            color = .systemRed
            size = 0.03
        case .info:
            color = .systemBlue
            size = 0.02
        case .question:
            color = .systemPurple
            size = 0.025
        }
        
        let mesh = MeshResource.generateSphere(radius: size)
        let material = SimpleMaterial(color: color.withAlphaComponent(0.8), isMetallic: false)
        
        let modelEntity = ModelEntity(mesh: mesh, materials: [material])
        entity.addChild(modelEntity)
        
        // Add pulsing animation
        let pulseAnimation = AnimationResource.makeTransform(
            duration: 2.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 1.0, value: Transform(scale: SIMD3<Float>(1.2, 1.2, 1.2))),
                .init(time: 2.0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        entity.playAnimation(pulseAnimation)
        
        return entity
    }
    
    private func createAnnotationText(_ annotation: ARAnnotation) -> Entity {
        let textMesh = MeshResource.generateText(
            annotation.text,
            extrusionDepth: 0.001,
            font: .systemFont(ofSize: 0.04),
            containerFrame: CGRect(x: 0, y: 0, width: 0.3, height: 0.1),
            alignment: .center,
            lineBreakMode: .byWordWrapping
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        
        // Add background panel
        let bgMesh = MeshResource.generatePlane(width: 0.32, depth: 0.08, cornerRadius: 0.01)
        let bgMaterial = SimpleMaterial(color: .black.withAlphaComponent(0.7), isMetallic: false)
        let bgEntity = ModelEntity(mesh: bgMesh, materials: [bgMaterial])
        bgEntity.position = [0, 0, -0.005]
        
        let containerEntity = Entity()
        containerEntity.addChild(bgEntity)
        containerEntity.addChild(textEntity)
        
        return containerEntity
    }
    
    private func createSharedDocumentVisualization(_ document: SharedDocument) {
        guard let arView = arView else { return }
        
        let documentAnchor = SharedAnchorEntity(document: document)
        documentAnchor.position = [0, 0, -1.5]
        
        // Create document representation
        let docVisual = createDocumentVisual(document)
        documentAnchor.addChild(docVisual)
        
        arView.scene.addAnchor(documentAnchor)
        sharedAnchors[document.id] = documentAnchor
    }
    
    private func createDocumentVisual(_ document: SharedDocument) -> Entity {
        let containerEntity = Entity()
        
        // Create document plane
        let mesh = MeshResource.generatePlane(width: 0.4, depth: 0.6)
        let material = SimpleMaterial(color: .white.withAlphaComponent(0.9), isMetallic: false)
        
        let documentEntity = ModelEntity(mesh: mesh, materials: [material])
        containerEntity.addChild(documentEntity)
        
        // Add title
        let titleMesh = MeshResource.generateText(
            document.title,
            extrusionDepth: 0.002,
            font: .boldSystemFont(ofSize: 0.06),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let titleMaterial = SimpleMaterial(color: .black, isMetallic: false)
        let titleEntity = ModelEntity(mesh: titleMesh, materials: [titleMaterial])
        titleEntity.position = [0, 0.25, 0.01]
        
        containerEntity.addChild(titleEntity)
        
        return containerEntity
    }
    
    private func createPointerEntity(_ pointer: ARPointer) {
        guard let arView = arView else { return }
        
        let pointerEntity = Entity()
        pointerEntity.position = pointer.position
        
        // Create pointer visual (arrow or hand)
        let pointerMesh = MeshResource.generateSphere(radius: 0.01)
        let pointerMaterial = SimpleMaterial(color: .systemRed, isMetallic: true)
        
        let pointerModel = ModelEntity(mesh: pointerMesh, materials: [pointerMaterial])
        pointerEntity.addChild(pointerModel)
        
        // Create pointing beam
        let beamMesh = MeshResource.generateCylinder(height: 0.1, radius: 0.002)
        let beamMaterial = SimpleMaterial(color: .systemRed.withAlphaComponent(0.6), isMetallic: false)
        
        let beamEntity = ModelEntity(mesh: beamMesh, materials: [beamMaterial])
        beamEntity.position = [0, -0.05, 0]
        pointerEntity.addChild(beamEntity)
        
        // Add to scene
        let anchorEntity = AnchorEntity(world: pointer.position)
        anchorEntity.addChild(pointerEntity)
        arView.scene.addAnchor(anchorEntity)
        
        // Auto-remove after duration
        DispatchQueue.main.asyncAfter(deadline: .now() + pointer.duration) {
            anchorEntity.removeFromParent()
        }
    }
    
    private func updateAnnotationWithComment(_ annotation: ARAnnotation, comment: ARComment) {
        guard let anchorEntity = sharedAnchors[annotation.id] else { return }
        
        // Add comment bubble
        let commentEntity = createCommentBubble(comment)
        commentEntity.position = [0.1, -0.05, 0]
        
        anchorEntity.addChild(commentEntity)
    }
    
    private func createCommentBubble(_ comment: ARComment) -> Entity {
        let bubbleEntity = Entity()
        
        // Create bubble background
        let bgMesh = MeshResource.generateSphere(radius: 0.03)
        let bgMaterial = SimpleMaterial(color: .systemBlue.withAlphaComponent(0.8), isMetallic: false)
        
        let bgEntity = ModelEntity(mesh: bgMesh, materials: [bgMaterial])
        bubbleEntity.addChild(bgEntity)
        
        // Add comment text (shortened)
        let previewText = String(comment.text.prefix(20)) + "..."
        let textMesh = MeshResource.generateText(
            previewText,
            extrusionDepth: 0.001,
            font: .systemFont(ofSize: 0.02),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        textEntity.position = [0, 0, 0.03]
        
        bubbleEntity.addChild(textEntity)
        
        return bubbleEntity
    }
    
    private func startSpatialAudio() {
        guard let audioEngine = audioEngine else { return }
        
        do {
            try audioEngine.start()
            print("Spatial audio started")
        } catch {
            print("Failed to start spatial audio: \(error)")
        }
    }
    
    private func stopSpatialAudio() {
        audioEngine?.stop()
    }
    
    private func updateSpatialAudioPositions() {
        guard let arView = arView,
              let mixer = spatialAudioMixer else { return }
        
        let cameraTransform = arView.session.currentFrame?.camera.transform ?? matrix_identity_float4x4
        
        // Update listener position
        mixer.listenerPosition = AVAudio3DPoint(x: 0, y: 0, z: 0)
        mixer.listenerAngularOrientation = AVAudio3DAngularOrientation(
            yaw: 0, pitch: 0, roll: 0
        )
        
        // Update peer audio positions
        for (peerID, userAnchor) in remoteUserAnchors {
            if let audioPlayer = voicePlayer[peerID] {
                let position = userAnchor.position
                // Update audio player 3D position based on user anchor
            }
        }
    }
    
    private func startVoiceRecording() {
        let audioSession = AVAudioSession.sharedInstance()
        
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default)
            try audioSession.setActive(true)
            
            let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioURL = documentsPath.appendingPathComponent("voice_message.m4a")
            
            let settings = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 12000,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            voiceRecorder = try AVAudioRecorder(url: audioURL, settings: settings)
            voiceRecorder?.record()
            
        } catch {
            print("Voice recording failed: \(error)")
        }
    }
    
    private func finishVoiceRecording() {
        guard let recorder = voiceRecorder else { return }
        
        recorder.stop()
        
        // Send audio file to connected peers
        if let audioData = try? Data(contentsOf: recorder.url) {
            let voiceMessage = VoiceMessage(
                id: UUID(),
                authorId: peerID,
                audioData: audioData,
                timestamp: Date(),
                duration: recorder.currentTime
            )
            
            broadcastVoiceMessage(voiceMessage)
        }
        
        voiceRecorder = nil
    }
    
    private func enableRemotePointing() {
        // Enable enhanced pointing capabilities for remote assistance
    }
    
    private func enableVoiceGuidance() {
        // Enable voice guidance features
    }
    
    private func clearSharedContent() {
        for (_, anchor) in sharedAnchors {
            anchor.removeFromParent()
        }
        sharedAnchors.removeAll()
        
        for (_, anchor) in remoteUserAnchors {
            anchor.removeFromParent()
        }
        remoteUserAnchors.removeAll()
        
        sharedAnnotations.removeAll()
        activeComments.removeAll()
    }
    
    // MARK: - Broadcast Methods
    
    private func broadcastAnnotation(_ annotation: ARAnnotation) {
        guard let session = mcSession,
              let data = try? JSONEncoder().encode(annotation) else { return }
        
        let message = CollaborationMessage(type: .annotation, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func broadcastComment(_ comment: ARComment) {
        guard let data = try? JSONEncoder().encode(comment) else { return }
        
        let message = CollaborationMessage(type: .comment, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func broadcastSharedDocument(_ document: SharedDocument) {
        guard let data = try? JSONEncoder().encode(document) else { return }
        
        let message = CollaborationMessage(type: .sharedDocument, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func broadcastPointer(_ pointer: ARPointer) {
        guard let data = try? JSONEncoder().encode(pointer) else { return }
        
        let message = CollaborationMessage(type: .pointer, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func broadcastVoiceMessage(_ voiceMessage: VoiceMessage) {
        guard let data = try? JSONEncoder().encode(voiceMessage) else { return }
        
        let message = CollaborationMessage(type: .voiceMessage, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func sendMessage(_ message: CollaborationMessage) {
        guard let session = mcSession,
              let messageData = try? JSONEncoder().encode(message) else { return }
        
        do {
            try session.send(messageData, toPeers: session.connectedPeers, with: .reliable)
        } catch {
            print("Failed to send message: \(error)")
        }
    }
    
    private func sendRemoteAssistanceRequest(_ request: RemoteAssistanceRequest, to peer: MCPeerID) {
        guard let data = try? JSONEncoder().encode(request) else { return }
        
        let message = CollaborationMessage(type: .remoteAssistanceRequest, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func sendRemoteAssistanceResponse(_ response: RemoteAssistanceResponse, to peer: MCPeerID) {
        guard let data = try? JSONEncoder().encode(response) else { return }
        
        let message = CollaborationMessage(type: .remoteAssistanceResponse, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func notifyPeersScreenSharingStarted() {
        let notification = ScreenSharingNotification(started: true, peerId: peerID, timestamp: Date())
        guard let data = try? JSONEncoder().encode(notification) else { return }
        
        let message = CollaborationMessage(type: .screenSharing, data: data, senderId: peerID)
        sendMessage(message)
    }
    
    private func notifyPeersScreenSharingStopped() {
        let notification = ScreenSharingNotification(started: false, peerId: peerID, timestamp: Date())
        guard let data = try? JSONEncoder().encode(notification) else { return }
        
        let message = CollaborationMessage(type: .screenSharing, data: data, senderId: peerID)
        sendMessage(message)
    }
}

// MARK: - MCSessionDelegate

extension ARCollaborationManager: MCSessionDelegate {
    public func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        DispatchQueue.main.async {
            switch state {
            case .connected:
                let peer = CollaborationPeer(id: peerID, displayName: peerID.displayName, state: .connected)
                self.connectedPeers.append(peer)
                print("Connected to peer: \(peerID.displayName)")
                
            case .connecting:
                print("Connecting to peer: \(peerID.displayName)")
                
            case .notConnected:
                self.connectedPeers.removeAll { $0.id == peerID }
                self.remoteUserAnchors[peerID]?.removeFromParent()
                self.remoteUserAnchors.removeValue(forKey: peerID)
                print("Disconnected from peer: \(peerID.displayName)")
                
            @unknown default:
                break
            }
        }
    }
    
    public func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        guard let message = try? JSONDecoder().decode(CollaborationMessage.self, from: data) else { return }
        
        DispatchQueue.main.async {
            self.handleReceivedMessage(message, from: peerID)
        }
    }
    
    public func session(_ session: MCSession, didReceive stream: InputStream, withName streamName: String, fromPeer peerID: MCPeerID) {
        // Handle incoming streams (e.g., screen sharing)
    }
    
    public func session(_ session: MCSession, didStartReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, with progress: Progress) {
        // Handle resource transfer progress
    }
    
    public func session(_ session: MCSession, didFinishReceivingResourceWithName resourceName: String, fromPeer peerID: MCPeerID, at localURL: URL?, withError error: Error?) {
        // Handle completed resource transfers
    }
    
    private func handleReceivedMessage(_ message: CollaborationMessage, from peerID: MCPeerID) {
        switch message.type {
        case .annotation:
            if let annotation = try? JSONDecoder().decode(ARAnnotation.self, from: message.data) {
                sharedAnnotations[annotation.id] = annotation
                createAnnotationEntity(annotation)
            }
            
        case .comment:
            if let comment = try? JSONDecoder().decode(ARComment.self, from: message.data),
               let annotation = sharedAnnotations[comment.annotationId] {
                activeComments[comment.id] = comment
                updateAnnotationWithComment(annotation, comment: comment)
            }
            
        case .sharedDocument:
            if let document = try? JSONDecoder().decode(SharedDocument.self, from: message.data) {
                createSharedDocumentVisualization(document)
            }
            
        case .pointer:
            if let pointer = try? JSONDecoder().decode(ARPointer.self, from: message.data) {
                createPointerEntity(pointer)
            }
            
        case .voiceMessage:
            if let voiceMessage = try? JSONDecoder().decode(VoiceMessage.self, from: message.data) {
                playVoiceMessage(voiceMessage, from: peerID)
            }
            
        case .remoteAssistanceRequest:
            if let request = try? JSONDecoder().decode(RemoteAssistanceRequest.self, from: message.data) {
                handleRemoteAssistanceRequest(request)
            }
            
        case .remoteAssistanceResponse:
            if let response = try? JSONDecoder().decode(RemoteAssistanceResponse.self, from: message.data) {
                handleRemoteAssistanceResponse(response)
            }
            
        case .screenSharing:
            if let notification = try? JSONDecoder().decode(ScreenSharingNotification.self, from: message.data) {
                handleScreenSharingNotification(notification, from: peerID)
            }
        }
    }
    
    private func playVoiceMessage(_ voiceMessage: VoiceMessage, from peerID: MCPeerID) {
        // Create temporary file and play audio with spatial positioning
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("received_voice.m4a")
        
        do {
            try voiceMessage.audioData.write(to: tempURL)
            let audioPlayer = try AVAudioPlayer(contentsOf: tempURL)
            
            voicePlayer[peerID] = audioPlayer
            audioPlayer.play()
            
            // Apply spatial audio positioning if available
            if let userAnchor = remoteUserAnchors[peerID] {
                // Position audio in 3D space based on user's position
            }
            
        } catch {
            print("Failed to play voice message: \(error)")
        }
    }
    
    private func handleRemoteAssistanceRequest(_ request: RemoteAssistanceRequest) {
        // Show UI to accept/decline remote assistance
        print("Remote assistance requested by \(request.requesterId.displayName)")
    }
    
    private func handleRemoteAssistanceResponse(_ response: RemoteAssistanceResponse) {
        if response.accepted {
            remoteAssistanceActive = true
            print("Remote assistance accepted by \(response.assistantId.displayName)")
        } else {
            print("Remote assistance declined by \(response.assistantId.displayName)")
        }
    }
    
    private func handleScreenSharingNotification(_ notification: ScreenSharingNotification, from peerID: MCPeerID) {
        if notification.started {
            // Create screen sharing visualization
            print("Screen sharing started by \(peerID.displayName)")
        } else {
            // Remove screen sharing visualization
            remoteScreens[peerID]?.removeFromParent()
            remoteScreens.removeValue(forKey: peerID)
            print("Screen sharing stopped by \(peerID.displayName)")
        }
    }
}

// MARK: - MCNearbyServiceAdvertiserDelegate

extension ARCollaborationManager: MCNearbyServiceAdvertiserDelegate {
    public func advertiser(_ advertiser: MCNearbyServiceAdvertiser, didReceiveInvitationFromPeer peerID: MCPeerID, withContext context: Data?, invitationHandler: @escaping (Bool, MCSession?) -> Void) {
        
        // Auto-accept invitations for now - in production, show UI
        invitationHandler(true, mcSession)
    }
}

// MARK: - MCNearbyServiceBrowserDelegate

extension ARCollaborationManager: MCNearbyServiceBrowserDelegate {
    public func browser(_ browser: MCNearbyServiceBrowser, foundPeer peerID: MCPeerID, withDiscoveryInfo info: [String : String]?) {
        
        // Auto-invite found peers - in production, show UI
        browser.invitePeer(peerID, to: mcSession!, withContext: nil, timeout: 10)
    }
    
    public func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {
        print("Lost peer: \(peerID.displayName)")
    }
}

// MARK: - ARSessionDelegate

extension ARCollaborationManager: ARSessionDelegate {
    public func session(_ session: ARSession, didOutputCollaborationData data: ARSession.CollaborationData) {
        // Share ARKit collaboration data with connected peers
        guard let mcSession = mcSession else { return }
        
        do {
            try mcSession.send(data, toPeers: mcSession.connectedPeers, with: .reliable)
        } catch {
            print("Failed to send collaboration data: \(error)")
        }
    }
}

// MARK: - ScreenRecorderDelegate

extension ARCollaborationManager: ScreenRecorderDelegate {
    func screenRecorder(_ recorder: ScreenRecorder, didCaptureFrame frameData: Data) {
        // Broadcast screen frame to connected peers
        guard let mcSession = mcSession else { return }
        
        let screenFrame = ScreenFrame(data: frameData, timestamp: Date(), senderId: peerID)
        
        if let frameData = try? JSONEncoder().encode(screenFrame) {
            do {
                try mcSession.send(frameData, toPeers: mcSession.connectedPeers, with: .unreliable)
            } catch {
                print("Failed to send screen frame: \(error)")
            }
        }
    }
    
    func screenRecorder(_ recorder: ScreenRecorder, didFailWithError error: Error) {
        print("Screen recording failed: \(error)")
        disableScreenSharing()
    }
}

// MARK: - Supporting Types

public enum CollaborationMode {
    case host
    case viewer
}

public struct CollaborationPeer: Identifiable {
    public let id: MCPeerID
    public let displayName: String
    public let state: MCSessionState
}

public enum ARAnnotationType: Codable {
    case highlight
    case warning
    case error
    case info
    case question
}

public struct ARAnnotation: Codable, Identifiable {
    public let id: UUID
    public let position: SIMD3<Float>
    public let text: String
    public let type: ARAnnotationType
    public let authorId: MCPeerID
    public let timestamp: Date
}

public struct ARComment: Codable, Identifiable {
    public let id: UUID
    public let annotationId: UUID
    public let text: String
    public let authorId: MCPeerID
    public let timestamp: Date
}

public struct ARPointer: Codable, Identifiable {
    public let id: UUID
    public let position: SIMD3<Float>
    public let authorId: MCPeerID
    public let timestamp: Date
    public let duration: TimeInterval
}

public struct SharedDocument: Codable, Identifiable {
    public let id: UUID
    public let title: String
    public let clauses: [ClauseData]
    public let sharedBy: MCPeerID
    public let timestamp: Date
}

public struct VoiceMessage: Codable, Identifiable {
    public let id: UUID
    public let authorId: MCPeerID
    public let audioData: Data
    public let timestamp: Date
    public let duration: TimeInterval
}

public struct RemoteAssistanceRequest: Codable {
    public let requesterId: MCPeerID
    public let targetId: MCPeerID
    public let timestamp: Date
}

public struct RemoteAssistanceResponse: Codable {
    public let accepted: Bool
    public let assistantId: MCPeerID
    public let requesterId: MCPeerID
    public let timestamp: Date
}

public struct ScreenSharingNotification: Codable {
    public let started: Bool
    public let peerId: MCPeerID
    public let timestamp: Date
}

public struct ScreenFrame: Codable {
    public let data: Data
    public let timestamp: Date
    public let senderId: MCPeerID
}

private enum CollaborationMessageType: Codable {
    case annotation
    case comment
    case sharedDocument
    case pointer
    case voiceMessage
    case remoteAssistanceRequest
    case remoteAssistanceResponse
    case screenSharing
}

private struct CollaborationMessage: Codable {
    let type: CollaborationMessageType
    let data: Data
    let senderId: MCPeerID
}

// MARK: - Custom Entity Classes

private class SharedAnchorEntity: Entity, HasAnchoring {
    let annotation: ARAnnotation?
    let document: SharedDocument?
    
    init(annotation: ARAnnotation) {
        self.annotation = annotation
        self.document = nil
        super.init()
    }
    
    init(document: SharedDocument) {
        self.annotation = nil
        self.document = document
        super.init()
    }
    
    required init() {
        self.annotation = nil
        self.document = nil
        super.init()
    }
}

private class UserAnchorEntity: Entity, HasAnchoring {
    let peerId: MCPeerID
    
    init(peerId: MCPeerID) {
        self.peerId = peerId
        super.init()
    }
    
    required init() {
        fatalError("init() has not been implemented")
    }
}

// MARK: - Screen Recording

private protocol ScreenRecorderDelegate: AnyObject {
    func screenRecorder(_ recorder: ScreenRecorder, didCaptureFrame frameData: Data)
    func screenRecorder(_ recorder: ScreenRecorder, didFailWithError error: Error)
}

private class ScreenRecorder {
    weak var delegate: ScreenRecorderDelegate?
    private var isRecording = false
    
    func startRecording() {
        guard !isRecording else { return }
        isRecording = true
        // Implement screen recording using RPScreenRecorder or similar
    }
    
    func stopRecording() {
        isRecording = false
    }
}

// MARK: - MCPeerID Codable Extension

extension MCPeerID: Codable {
    enum CodingKeys: String, CodingKey {
        case displayName
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(displayName, forKey: .displayName)
    }
    
    public convenience init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let displayName = try container.decode(String.self, forKey: .displayName)
        self.init(displayName: displayName)
    }
}

// MARK: - SIMD3 Codable Extension

extension SIMD3: Codable where Scalar: Codable {
    public func encode(to encoder: Encoder) throws {
        var container = encoder.unkeyedContainer()
        try container.encode(x)
        try container.encode(y)
        try container.encode(z)
    }
    
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        let x = try container.decode(Scalar.self)
        let y = try container.decode(Scalar.self)
        let z = try container.decode(Scalar.self)
        self.init(x, y, z)
    }
}