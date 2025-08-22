//
//  AIAssistant.swift
//  AR Document Analysis Framework
//
//  AR AI assistant with avatar, contextual help, and intelligent recommendations
//

import ARKit
import RealityKit
import SwiftUI
import Combine
import Speech
import AVFoundation
import NaturalLanguage

@available(iOS 13.0, *)
public class AIAssistant: NSObject, ObservableObject {
    
    // MARK: - Properties
    
    @Published public var isAssistantActive = false
    @Published public var assistantMode: AssistantMode = .contextual
    @Published public var currentQuery: String = ""
    @Published public var isListening = false
    @Published public var isThinking = false
    @Published public var confidenceLevel: Float = 0.0
    @Published public var suggestions: [AISuggestion] = []
    @Published public var activeRecommendations: [AIRecommendation] = []
    
    private var arView: ARView?
    private var assistantAnchor: AnchorEntity?
    private var avatarEntity: AssistantAvatarEntity?
    private var helpBubbles: [UUID: HelpBubbleEntity] = [:]
    
    // Speech and NLP
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var audioEngine = AVAudioEngine()
    private var speechSynthesizer = AVSpeechSynthesizer()
    private var nlProcessor = NLLanguageRecognizer()
    
    // AI Context
    private var currentDocument: AnalyzedDocument?
    private var conversationHistory: [ConversationMessage] = []
    private var userPreferences: UserPreferences = UserPreferences()
    private var knowledgeBase: AIKnowledgeBase = AIKnowledgeBase()
    
    // Visual search and analysis
    private var visualSearchActive = false
    private var contextualOverlays: [UUID: Entity] = [:]
    
    // Animation controllers
    private var animationControllers: [AnimationPlaybackController] = []
    
    // MARK: - Initialization
    
    public override init() {
        super.init()
        setupSpeechRecognition()
        setupKnowledgeBase()
        speechSynthesizer.delegate = self
    }
    
    deinit {
        deactivateAssistant()
    }
    
    // MARK: - Public Methods
    
    public func activateAssistant(in arView: ARView, with document: AnalyzedDocument? = nil) {
        self.arView = arView
        self.currentDocument = document
        
        createAssistantAvatar()
        setupContextualHelp()
        
        isAssistantActive = true
        
        // Welcome message
        speakResponse("Hello! I'm your AR document analysis assistant. How can I help you today?")
        
        startContextualAnalysis()
    }
    
    public func deactivateAssistant() {
        stopListening()
        clearVisualizations()
        
        isAssistantActive = false
        assistantAnchor?.removeFromParent()
        assistantAnchor = nil
        avatarEntity = nil
        
        // Stop all animations
        animationControllers.forEach { $0.stop() }
        animationControllers.removeAll()
    }
    
    public func askQuestion(_ question: String) {
        currentQuery = question
        addToConversationHistory(.user(question))
        
        showThinkingIndicator()
        
        Task {
            let response = await processQuery(question)
            await MainActor.run {
                self.hideThinkingIndicator()
                self.provideResponse(response)
            }
        }
    }
    
    public func startListening() {
        guard !isListening else { return }
        
        requestSpeechAuthorization { [weak self] authorized in
            guard authorized else { return }
            
            DispatchQueue.main.async {
                self?.beginSpeechRecognition()
            }
        }
    }
    
    public func stopListening() {
        guard isListening else { return }
        
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        
        isListening = false
        updateAvatarState(.idle)
    }
    
    public func enableVisualSearch() {
        visualSearchActive = true
        setupVisualSearchOverlay()
        speakResponse("Visual search activated. Point your device at any part of the document to get contextual information.")
    }
    
    public func disableVisualSearch() {
        visualSearchActive = false
        clearVisualSearchOverlays()
    }
    
    public func highlightRisks() {
        guard let document = currentDocument else {
            speakResponse("Please scan a document first to analyze risks.")
            return
        }
        
        let highRiskClauses = document.clauses.filter { $0.riskLevel == .high || $0.riskLevel == .critical }
        
        for clause in highRiskClauses {
            createRiskHighlight(for: clause)
        }
        
        let riskCount = highRiskClauses.count
        speakResponse("I found \(riskCount) high-risk clauses. They are now highlighted in your view.")
    }
    
    public func explainClause(_ clauseId: UUID) {
        guard let document = currentDocument,
              let clause = document.clauses.first(where: { $0.id == clauseId }) else {
            speakResponse("I couldn't find that clause.")
            return
        }
        
        showThinkingIndicator()
        
        Task {
            let explanation = await generateClauseExplanation(clause)
            await MainActor.run {
                self.hideThinkingIndicator()
                self.showExplanationBubble(explanation, for: clause)
                self.speakResponse(explanation.summary)
            }
        }
    }
    
    public func suggestImprovements() {
        guard let document = currentDocument else {
            speakResponse("Please scan a document first to get improvement suggestions.")
            return
        }
        
        showThinkingIndicator()
        
        Task {
            let improvements = await generateImprovementSuggestions(for: document)
            await MainActor.run {
                self.hideThinkingIndicator()
                self.activeRecommendations = improvements
                self.showImprovementSuggestions(improvements)
                
                let count = improvements.count
                self.speakResponse("I have \(count) suggestions to improve your document. Check the highlighted areas.")
            }
        }
    }
    
    public func compareWithPrecedents() {
        guard let document = currentDocument else {
            speakResponse("Please scan a document first to compare with precedents.")
            return
        }
        
        showThinkingIndicator()
        
        Task {
            let precedentAnalysis = await analyzeAgainstPrecedents(document)
            await MainActor.run {
                self.hideThinkingIndicator()
                self.showPrecedentAnalysis(precedentAnalysis)
                self.speakResponse("I've analyzed your document against similar precedents. Here's what I found.")
            }
        }
    }
    
    public func enablePredictiveHighlighting() {
        startPredictiveAnalysis()
        speakResponse("Predictive highlighting is now active. I'll highlight important sections as you review the document.")
    }
    
    public func enableTranslationOverlay() {
        guard let document = currentDocument else { return }
        
        showThinkingIndicator()
        
        Task {
            let translations = await translateDocument(document)
            await MainActor.run {
                self.hideThinkingIndicator()
                self.showTranslationOverlay(translations)
                self.speakResponse("Translation overlay is now active. Tap on any text to see translations.")
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func setupSpeechRecognition() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
        speechRecognizer?.delegate = self
    }
    
    private func setupKnowledgeBase() {
        knowledgeBase.loadLegalTerms()
        knowledgeBase.loadContractPatterns()
        knowledgeBase.loadRiskPatterns()
    }
    
    private func createAssistantAvatar() {
        guard let arView = arView else { return }
        
        // Position avatar to the right of the user's view
        let cameraTransform = arView.session.currentFrame?.camera.transform ?? matrix_identity_float4x4
        let anchorEntity = AnchorEntity(world: cameraTransform)
        anchorEntity.position = [0.8, 0, -1.2]
        
        avatarEntity = AssistantAvatarEntity()
        avatarEntity!.position = [0, 0, 0]
        
        createAvatarVisuals()
        
        anchorEntity.addChild(avatarEntity!)
        arView.scene.addAnchor(anchorEntity)
        
        self.assistantAnchor = anchorEntity
        
        // Start idle animation
        startAvatarIdleAnimation()
    }
    
    private func createAvatarVisuals() {
        guard let avatarEntity = avatarEntity else { return }
        
        // Create holographic assistant representation
        let bodyMesh = MeshResource.generateSphere(radius: 0.15)
        let bodyMaterial = createHologramMaterial(color: .systemBlue)
        
        let bodyEntity = ModelEntity(mesh: bodyMesh, materials: [bodyMaterial])
        avatarEntity.addChild(bodyEntity)
        
        // Create floating elements around avatar
        createAvatarAura()
        
        // Add face/expression area
        let faceMesh = MeshResource.generateSphere(radius: 0.08)
        let faceMaterial = createHologramMaterial(color: .systemCyan)
        
        let faceEntity = ModelEntity(mesh: faceMesh, materials: [faceMaterial])
        faceEntity.position = [0, 0.05, 0.1]
        avatarEntity.addChild(faceEntity)
        
        avatarEntity.bodyEntity = bodyEntity
        avatarEntity.faceEntity = faceEntity
    }
    
    private func createHologramMaterial(color: UIColor) -> Material {
        var material = SimpleMaterial()
        material.baseColor = MaterialColorParameter.color(color.withAlphaComponent(0.7))
        material.emissiveColor = MaterialColorParameter.color(color.withAlphaComponent(0.3))
        material.metallic = 0.8
        material.roughness = 0.2
        return material
    }
    
    private func createAvatarAura() {
        guard let avatarEntity = avatarEntity else { return }
        
        // Create floating particles around avatar
        for i in 0..<8 {
            let angle = Float(i) * Float.pi / 4
            let radius: Float = 0.25
            
            let x = cos(angle) * radius
            let z = sin(angle) * radius
            let y = Float.random(in: -0.1...0.1)
            
            let particleMesh = MeshResource.generateSphere(radius: 0.005)
            let particleMaterial = createHologramMaterial(color: .systemTeal)
            
            let particleEntity = ModelEntity(mesh: particleMesh, materials: [particleMaterial])
            particleEntity.position = [x, y, z]
            
            avatarEntity.addChild(particleEntity)
            
            // Add floating animation
            let floatingAnimation = createFloatingAnimation(delay: Float(i) * 0.2)
            let controller = particleEntity.playAnimation(floatingAnimation)
            animationControllers.append(controller)
        }
    }
    
    private func createFloatingAnimation(delay: Float = 0) -> AnimationResource {
        return AnimationResource.makeTransform(
            duration: 3.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: true,
            bindTarget: .transform,
            keyframes: [
                .init(time: delay, value: Transform(translation: [0, 0, 0])),
                .init(time: delay + 1.5, value: Transform(translation: [0, 0.02, 0])),
                .init(time: delay + 3.0, value: Transform(translation: [0, 0, 0]))
            ]
        )
    }
    
    private func startAvatarIdleAnimation() {
        guard let bodyEntity = avatarEntity?.bodyEntity else { return }
        
        let pulseAnimation = AnimationResource.makeTransform(
            duration: 2.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 1.0, value: Transform(scale: SIMD3<Float>(1.05, 1.05, 1.05))),
                .init(time: 2.0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        let controller = bodyEntity.playAnimation(pulseAnimation)
        animationControllers.append(controller)
    }
    
    private func updateAvatarState(_ state: AvatarState) {
        guard let avatarEntity = avatarEntity,
              let bodyEntity = avatarEntity.bodyEntity else { return }
        
        let color: UIColor
        let animationDuration: TimeInterval
        
        switch state {
        case .idle:
            color = .systemBlue
            animationDuration = 2.0
        case .listening:
            color = .systemGreen
            animationDuration = 1.0
        case .thinking:
            color = .systemOrange
            animationDuration = 0.5
        case .speaking:
            color = .systemPurple
            animationDuration = 1.5
        }
        
        // Update material
        let newMaterial = createHologramMaterial(color: color)
        bodyEntity.model?.materials = [newMaterial]
        
        // Update animation speed
        updateAvatarAnimationSpeed(animationDuration)
    }
    
    private func updateAvatarAnimationSpeed(_ duration: TimeInterval) {
        // Update existing animations with new duration
        // This is a simplified version - full implementation would manage animation transitions
    }
    
    private func setupContextualHelp() {
        guard let document = currentDocument else { return }
        
        // Create help bubbles for complex clauses
        for clause in document.clauses where clause.riskLevel == .high || clause.riskLevel == .critical {
            createHelpBubble(for: clause)
        }
    }
    
    private func createHelpBubble(for clause: ClauseData) {
        guard let arView = arView else { return }
        
        let position = calculateClausePosition(clause)
        let helpBubble = HelpBubbleEntity(clause: clause)
        helpBubble.position = position + [0, 0.1, 0]
        
        createHelpBubbleVisuals(helpBubble)
        
        let anchorEntity = AnchorEntity(world: position)
        anchorEntity.addChild(helpBubble)
        arView.scene.addAnchor(anchorEntity)
        
        helpBubbles[clause.id] = helpBubble
    }
    
    private func createHelpBubbleVisuals(_ helpBubble: HelpBubbleEntity) {
        // Create question mark or info icon
        let iconMesh = MeshResource.generateSphere(radius: 0.02)
        let iconMaterial = SimpleMaterial(color: .systemBlue.withAlphaComponent(0.8), isMetallic: false)
        
        let iconEntity = ModelEntity(mesh: iconMesh, materials: [iconMaterial])
        iconEntity.components.set(InputTargetComponent())
        
        helpBubble.addChild(iconEntity)
        
        // Add pulsing animation
        let pulseAnimation = AnimationResource.makeTransform(
            duration: 1.5,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 0.75, value: Transform(scale: SIMD3<Float>(1.3, 1.3, 1.3))),
                .init(time: 1.5, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        let controller = helpBubble.playAnimation(pulseAnimation)
        animationControllers.append(controller)
    }
    
    private func calculateClausePosition(_ clause: ClauseData) -> SIMD3<Float> {
        // Calculate position based on clause properties
        let sectionHash = Float(clause.section.hashValue % 360) * .pi / 180
        let orderOffset = Float(clause.order) * 0.1
        
        let x = cos(sectionHash) * (0.4 + orderOffset * 0.05)
        let y = Float(clause.riskLevel.rawValue - 1) * 0.15
        let z = sin(sectionHash) * (0.4 + orderOffset * 0.05)
        
        return SIMD3<Float>(x, y, z)
    }
    
    private func requestSpeechAuthorization(completion: @escaping (Bool) -> Void) {
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                completion(status == .authorized)
            }
        }
    }
    
    private func beginSpeechRecognition() {
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
            isListening = true
            updateAvatarState(.listening)
            
            recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
                if let result = result {
                    let transcribedText = result.bestTranscription.formattedString
                    
                    DispatchQueue.main.async {
                        self?.currentQuery = transcribedText
                        
                        if result.isFinal {
                            self?.stopListening()
                            self?.askQuestion(transcribedText)
                        }
                    }
                }
                
                if let error = error {
                    DispatchQueue.main.async {
                        self?.stopListening()
                        print("Speech recognition error: \(error)")
                    }
                }
            }
            
        } catch {
            print("Audio engine start error: \(error)")
            isListening = false
        }
    }
    
    private func processQuery(_ query: String) async -> AIResponse {
        let intent = analyzeUserIntent(query)
        
        switch intent.type {
        case .explain:
            return await handleExplainIntent(intent)
        case .analyze:
            return await handleAnalyzeIntent(intent)
        case .suggest:
            return await handleSuggestIntent(intent)
        case .compare:
            return await handleCompareIntent(intent)
        case .search:
            return await handleSearchIntent(intent)
        case .general:
            return await handleGeneralIntent(intent)
        }
    }
    
    private func analyzeUserIntent(_ query: String) -> UserIntent {
        let lowercaseQuery = query.lowercased()
        
        // Simple intent classification - in production, use ML model
        if lowercaseQuery.contains("explain") || lowercaseQuery.contains("what is") || lowercaseQuery.contains("mean") {
            return UserIntent(type: .explain, entities: extractEntities(from: query), confidence: 0.8)
        } else if lowercaseQuery.contains("analyze") || lowercaseQuery.contains("check") || lowercaseQuery.contains("review") {
            return UserIntent(type: .analyze, entities: extractEntities(from: query), confidence: 0.85)
        } else if lowercaseQuery.contains("suggest") || lowercaseQuery.contains("improve") || lowercaseQuery.contains("recommend") {
            return UserIntent(type: .suggest, entities: extractEntities(from: query), confidence: 0.9)
        } else if lowercaseQuery.contains("compare") || lowercaseQuery.contains("similar") || lowercaseQuery.contains("precedent") {
            return UserIntent(type: .compare, entities: extractEntities(from: query), confidence: 0.7)
        } else if lowercaseQuery.contains("find") || lowercaseQuery.contains("search") || lowercaseQuery.contains("look for") {
            return UserIntent(type: .search, entities: extractEntities(from: query), confidence: 0.8)
        } else {
            return UserIntent(type: .general, entities: [], confidence: 0.6)
        }
    }
    
    private func extractEntities(from query: String) -> [String] {
        // Extract relevant entities (clause types, legal terms, etc.)
        var entities: [String] = []
        
        // Use NLLanguageRecognizer for named entity recognition
        nlProcessor.processString(query)
        
        // For simplicity, return keywords - full implementation would use NLP
        let keywords = ["arbitration", "termination", "liability", "indemnification", "warranty", "breach"]
        
        for keyword in keywords {
            if query.lowercased().contains(keyword) {
                entities.append(keyword)
            }
        }
        
        return entities
    }
    
    private func handleExplainIntent(_ intent: UserIntent) async -> AIResponse {
        guard let document = currentDocument else {
            return AIResponse(text: "I don't have a document loaded to explain. Please scan a document first.", confidence: 1.0, actions: [])
        }
        
        if intent.entities.isEmpty {
            return AIResponse(
                text: "I can explain various parts of your document. What specifically would you like me to explain?",
                confidence: 0.8,
                actions: [.showClauseList]
            )
        }
        
        // Find relevant clauses based on entities
        let relevantClauses = document.clauses.filter { clause in
            intent.entities.contains { entity in
                clause.content.lowercased().contains(entity.lowercased()) ||
                clause.title.lowercased().contains(entity.lowercased())
            }
        }
        
        if relevantClauses.isEmpty {
            return AIResponse(
                text: "I couldn't find any clauses related to \(intent.entities.joined(separator: ", ")) in your document.",
                confidence: 0.7,
                actions: []
            )
        }
        
        let explanation = await generateExplanation(for: relevantClauses.first!)
        return AIResponse(text: explanation, confidence: 0.9, actions: [.highlightClause(relevantClauses.first!.id)])
    }
    
    private func handleAnalyzeIntent(_ intent: UserIntent) async -> AIResponse {
        guard let document = currentDocument else {
            return AIResponse(text: "Please scan a document first for me to analyze.", confidence: 1.0, actions: [])
        }
        
        let analysis = await performDocumentAnalysis(document)
        return AIResponse(text: analysis, confidence: 0.9, actions: [.showRiskOverlay])
    }
    
    private func handleSuggestIntent(_ intent: UserIntent) async -> AIResponse {
        guard let document = currentDocument else {
            return AIResponse(text: "I need a document to analyze before I can make suggestions.", confidence: 1.0, actions: [])
        }
        
        let suggestions = await generateImprovementSuggestions(for: document)
        activeRecommendations = suggestions
        
        let suggestionText = "I have \(suggestions.count) suggestions for improving your document. The key areas are: " +
                           suggestions.prefix(3).map { $0.title }.joined(separator: ", ")
        
        return AIResponse(text: suggestionText, confidence: 0.9, actions: [.showSuggestions])
    }
    
    private func handleCompareIntent(_ intent: UserIntent) async -> AIResponse {
        guard let document = currentDocument else {
            return AIResponse(text: "Please provide a document to compare against precedents.", confidence: 1.0, actions: [])
        }
        
        let precedentAnalysis = await analyzeAgainstPrecedents(document)
        return AIResponse(text: precedentAnalysis.summary, confidence: 0.8, actions: [.showPrecedentComparison])
    }
    
    private func handleSearchIntent(_ intent: UserIntent) async -> AIResponse {
        guard let document = currentDocument else {
            return AIResponse(text: "I need a document loaded to search through.", confidence: 1.0, actions: [])
        }
        
        if intent.entities.isEmpty {
            return AIResponse(text: "What would you like me to search for in the document?", confidence: 0.8, actions: [])
        }
        
        let searchResults = searchDocument(document, for: intent.entities)
        
        if searchResults.isEmpty {
            return AIResponse(
                text: "I couldn't find any mentions of \(intent.entities.joined(separator: ", ")) in your document.",
                confidence: 0.8,
                actions: []
            )
        }
        
        return AIResponse(
            text: "I found \(searchResults.count) references to \(intent.entities.joined(separator: ", ")). Let me highlight them for you.",
            confidence: 0.9,
            actions: searchResults.map { .highlightClause($0.id) }
        )
    }
    
    private func handleGeneralIntent(_ intent: UserIntent) async -> AIResponse {
        // Handle general questions about document analysis, legal terms, etc.
        let response = await generateGeneralResponse(intent)
        return response
    }
    
    private func provideResponse(_ response: AIResponse) {
        addToConversationHistory(.assistant(response.text))
        speakResponse(response.text)
        
        // Execute actions
        for action in response.actions {
            executeAction(action)
        }
        
        confidenceLevel = response.confidence
    }
    
    private func executeAction(_ action: AIAction) {
        switch action {
        case .highlightClause(let clauseId):
            highlightClause(clauseId)
        case .showRiskOverlay:
            showRiskOverlay()
        case .showSuggestions:
            showSuggestionOverlays()
        case .showClauseList:
            showClauseList()
        case .showPrecedentComparison:
            showPrecedentComparison()
        }
    }
    
    private func speakResponse(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.0
        utterance.volume = 0.8
        
        updateAvatarState(.speaking)
        speechSynthesizer.speak(utterance)
    }
    
    private func showThinkingIndicator() {
        isThinking = true
        updateAvatarState(.thinking)
        
        // Create thinking bubble
        guard let avatarEntity = avatarEntity else { return }
        
        let thinkingBubble = createThinkingBubble()
        thinkingBubble.position = [0, 0.3, 0]
        avatarEntity.addChild(thinkingBubble)
        avatarEntity.thinkingBubble = thinkingBubble
    }
    
    private func hideThinkingIndicator() {
        isThinking = false
        updateAvatarState(.idle)
        
        avatarEntity?.thinkingBubble?.removeFromParent()
        avatarEntity?.thinkingBubble = nil
    }
    
    private func createThinkingBubble() -> Entity {
        let bubbleEntity = Entity()
        
        // Create bubble background
        let bubbleMesh = MeshResource.generateSphere(radius: 0.08)
        let bubbleMaterial = SimpleMaterial(color: .white.withAlphaComponent(0.9), isMetallic: false)
        
        let bubbleModel = ModelEntity(mesh: bubbleMesh, materials: [bubbleMaterial])
        bubbleEntity.addChild(bubbleModel)
        
        // Add dots animation
        for i in 0..<3 {
            let dotMesh = MeshResource.generateSphere(radius: 0.01)
            let dotMaterial = SimpleMaterial(color: .systemGray, isMetallic: false)
            
            let dotEntity = ModelEntity(mesh: dotMesh, materials: [dotMaterial])
            let xOffset = Float(i - 1) * 0.03
            dotEntity.position = [xOffset, 0, 0.08]
            
            bubbleEntity.addChild(dotEntity)
            
            // Animate dots
            let bounceAnimation = AnimationResource.makeTransform(
                duration: 1.0,
                repeatMode: .repeat,
                blendLayer: 0,
                additive: true,
                bindTarget: .transform,
                keyframes: [
                    .init(time: Float(i) * 0.2, value: Transform(translation: [xOffset, 0, 0.08])),
                    .init(time: Float(i) * 0.2 + 0.3, value: Transform(translation: [xOffset, 0.02, 0.08])),
                    .init(time: Float(i) * 0.2 + 0.6, value: Transform(translation: [xOffset, 0, 0.08]))
                ]
            )
            
            let controller = dotEntity.playAnimation(bounceAnimation)
            animationControllers.append(controller)
        }
        
        return bubbleEntity
    }
    
    // MARK: - Visual Search and Analysis
    
    private func setupVisualSearchOverlay() {
        guard let arView = arView else { return }
        
        // Create visual search indicator
        let searchOverlay = Entity()
        
        // Create crosshair or search reticle
        let reticleMesh = MeshResource.generatePlane(width: 0.05, depth: 0.05)
        let reticleMaterial = SimpleMaterial(color: .systemGreen.withAlphaComponent(0.8), isMetallic: false)
        
        let reticleEntity = ModelEntity(mesh: reticleMesh, materials: [reticleMaterial])
        searchOverlay.addChild(reticleEntity)
        
        // Position in center of view
        let cameraTransform = arView.session.currentFrame?.camera.transform ?? matrix_identity_float4x4
        let anchorEntity = AnchorEntity(world: cameraTransform)
        anchorEntity.position = [0, 0, -0.5]
        
        anchorEntity.addChild(searchOverlay)
        arView.scene.addAnchor(anchorEntity)
        
        contextualOverlays[UUID()] = anchorEntity
    }
    
    private func clearVisualSearchOverlays() {
        for (_, overlay) in contextualOverlays {
            overlay.removeFromParent()
        }
        contextualOverlays.removeAll()
    }
    
    private func startContextualAnalysis() {
        // Continuously analyze what the user is looking at
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] timer in
            guard let self = self, self.isAssistantActive else {
                timer.invalidate()
                return
            }
            
            self.analyzeCurrentView()
        }
    }
    
    private func analyzeCurrentView() {
        guard let arView = arView,
              let frame = arView.session.currentFrame else { return }
        
        // Analyze what the camera is looking at
        // This would involve computer vision to detect document sections
        // and provide contextual suggestions
    }
    
    private func startPredictiveAnalysis() {
        // Monitor user behavior and predict what they might need help with
        // This is a placeholder for ML-based predictive analysis
    }
    
    // MARK: - AI Processing Methods
    
    private func generateClauseExplanation(_ clause: ClauseData) async -> ClauseExplanation {
        // Simulate AI processing - in production, would call actual AI service
        await Task.sleep(nanoseconds: 2_000_000_000) // 2 second delay
        
        return ClauseExplanation(
            clauseId: clause.id,
            summary: "This \(clause.riskLevel.displayName.lowercased()) clause deals with \(clause.section.lowercased()). It establishes specific obligations and may require careful review.",
            details: clause.content,
            recommendations: generateRecommendations(for: clause),
            relatedClauses: []
        )
    }
    
    private func generateImprovementSuggestions(for document: AnalyzedDocument) async -> [AIRecommendation] {
        // Simulate AI processing
        await Task.sleep(nanoseconds: 1_500_000_000)
        
        var recommendations: [AIRecommendation] = []
        
        for clause in document.clauses where clause.riskLevel != .low {
            let recommendation = AIRecommendation(
                id: UUID(),
                clauseId: clause.id,
                type: .riskMitigation,
                title: "Review \(clause.section) clause",
                description: "This clause presents \(clause.riskLevel.displayName.lowercased()) risk and should be carefully reviewed.",
                priority: clause.riskLevel.rawValue,
                suggestedAction: "Consider adding protective language or seeking legal review."
            )
            recommendations.append(recommendation)
        }
        
        return recommendations
    }
    
    private func analyzeAgainstPrecedents(_ document: AnalyzedDocument) async -> PrecedentAnalysis {
        // Simulate precedent analysis
        await Task.sleep(nanoseconds: 2_500_000_000)
        
        return PrecedentAnalysis(
            summary: "Your document contains several clauses that differ from industry standards. I've identified 3 areas where similar agreements typically include additional protections.",
            similarities: [],
            differences: [],
            recommendations: []
        )
    }
    
    private func translateDocument(_ document: AnalyzedDocument) async -> [Translation] {
        // Simulate translation
        await Task.sleep(nanoseconds: 1_000_000_000)
        
        return document.clauses.map { clause in
            Translation(
                clauseId: clause.id,
                originalText: clause.content,
                translatedText: "Translated: " + clause.content,
                targetLanguage: "es",
                confidence: 0.85
            )
        }
    }
    
    private func generateExplanation(for clause: ClauseData) async -> String {
        await Task.sleep(nanoseconds: 1_000_000_000)
        
        return "This clause in the \(clause.section) section establishes important terms regarding \(clause.title.lowercased()). Based on my analysis, this is a \(clause.riskLevel.displayName.lowercased()) clause that requires attention."
    }
    
    private func performDocumentAnalysis(_ document: AnalyzedDocument) async -> String {
        await Task.sleep(nanoseconds: 2_000_000_000)
        
        let highRisk = document.clauses.filter { $0.riskLevel == .high || $0.riskLevel == .critical }.count
        let totalClauses = document.clauses.count
        
        return "I've analyzed your document and found \(totalClauses) clauses. \(highRisk) of these are high-risk areas that need attention. The main concerns are related to liability, termination, and dispute resolution."
    }
    
    private func generateGeneralResponse(_ intent: UserIntent) async -> AIResponse {
        await Task.sleep(nanoseconds: 1_000_000_000)
        
        return AIResponse(
            text: "I'm here to help you analyze documents, explain legal clauses, and identify potential risks. What would you like to know?",
            confidence: 0.8,
            actions: []
        )
    }
    
    private func searchDocument(_ document: AnalyzedDocument, for entities: [String]) -> [ClauseData] {
        return document.clauses.filter { clause in
            entities.contains { entity in
                clause.content.lowercased().contains(entity.lowercased()) ||
                clause.title.lowercased().contains(entity.lowercased())
            }
        }
    }
    
    private func generateRecommendations(for clause: ClauseData) -> [String] {
        var recommendations: [String] = []
        
        switch clause.riskLevel {
        case .critical:
            recommendations.append("Seek immediate legal counsel")
            recommendations.append("Consider rejecting this clause")
        case .high:
            recommendations.append("Request modifications to reduce risk")
            recommendations.append("Add protective language")
        case .medium:
            recommendations.append("Review carefully before accepting")
        case .low:
            recommendations.append("Standard clause - generally acceptable")
        }
        
        return recommendations
    }
    
    // MARK: - UI Helper Methods
    
    private func showExplanationBubble(_ explanation: ClauseExplanation, for clause: ClauseData) {
        // Create explanation bubble in AR space
        guard let arView = arView else { return }
        
        let position = calculateClausePosition(clause)
        let explanationEntity = createExplanationBubble(explanation)
        explanationEntity.position = position + [0, 0.2, 0]
        
        let anchorEntity = AnchorEntity(world: position)
        anchorEntity.addChild(explanationEntity)
        arView.scene.addAnchor(anchorEntity)
        
        contextualOverlays[explanation.clauseId] = anchorEntity
        
        // Auto-remove after 10 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
            anchorEntity.removeFromParent()
            self.contextualOverlays.removeValue(forKey: explanation.clauseId)
        }
    }
    
    private func createExplanationBubble(_ explanation: ClauseExplanation) -> Entity {
        let containerEntity = Entity()
        
        // Create background panel
        let bgMesh = MeshResource.generatePlane(width: 0.4, depth: 0.3, cornerRadius: 0.02)
        let bgMaterial = SimpleMaterial(color: .black.withAlphaComponent(0.8), isMetallic: false)
        
        let bgEntity = ModelEntity(mesh: bgMesh, materials: [bgMaterial])
        containerEntity.addChild(bgEntity)
        
        // Create text
        let textMesh = MeshResource.generateText(
            explanation.summary,
            extrusionDepth: 0.001,
            font: .systemFont(ofSize: 0.03),
            containerFrame: CGRect(x: 0, y: 0, width: 0.35, height: 0.25),
            alignment: .left,
            lineBreakMode: .byWordWrapping
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        textEntity.position = [0, 0, 0.01]
        
        containerEntity.addChild(textEntity)
        
        return containerEntity
    }
    
    private func highlightClause(_ clauseId: UUID) {
        // Highlight specific clause in AR
        createRiskHighlight(for: currentDocument?.clauses.first { $0.id == clauseId })
    }
    
    private func createRiskHighlight(for clause: ClauseData?) {
        guard let clause = clause, let arView = arView else { return }
        
        let position = calculateClausePosition(clause)
        let highlightEntity = createHighlightRing(color: clause.riskLevel.color)
        highlightEntity.position = position
        
        let anchorEntity = AnchorEntity(world: position)
        anchorEntity.addChild(highlightEntity)
        arView.scene.addAnchor(anchorEntity)
        
        contextualOverlays[clause.id] = anchorEntity
        
        // Auto-remove after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            anchorEntity.removeFromParent()
            self.contextualOverlays.removeValue(forKey: clause.id)
        }
    }
    
    private func createHighlightRing(color: UIColor) -> Entity {
        let ringEntity = Entity()
        
        let ringMesh = MeshResource.generatePlane(width: 0.15, depth: 0.15, cornerRadius: 0.075)
        let ringMaterial = SimpleMaterial(color: color.withAlphaComponent(0.6), isMetallic: false)
        
        let ringModel = ModelEntity(mesh: ringMesh, materials: [ringMaterial])
        ringEntity.addChild(ringModel)
        
        // Add pulsing animation
        let pulseAnimation = AnimationResource.makeTransform(
            duration: 1.0,
            repeatMode: .repeat,
            blendLayer: 0,
            additive: false,
            bindTarget: .transform,
            keyframes: [
                .init(time: 0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0))),
                .init(time: 0.5, value: Transform(scale: SIMD3<Float>(1.2, 1.2, 1.0))),
                .init(time: 1.0, value: Transform(scale: SIMD3<Float>(1.0, 1.0, 1.0)))
            ]
        )
        
        let controller = ringEntity.playAnimation(pulseAnimation)
        animationControllers.append(controller)
        
        return ringEntity
    }
    
    private func showRiskOverlay() {
        // Show risk overlay visualization
        guard let document = currentDocument else { return }
        
        for clause in document.clauses where clause.riskLevel != .low {
            createRiskHighlight(for: clause)
        }
    }
    
    private func showSuggestionOverlays() {
        // Show improvement suggestion overlays
        for recommendation in activeRecommendations {
            showRecommendationOverlay(recommendation)
        }
    }
    
    private func showRecommendationOverlay(_ recommendation: AIRecommendation) {
        guard let clause = currentDocument?.clauses.first(where: { $0.id == recommendation.clauseId }),
              let arView = arView else { return }
        
        let position = calculateClausePosition(clause)
        let suggestionEntity = createSuggestionBubble(recommendation)
        suggestionEntity.position = position + [0.1, 0.1, 0]
        
        let anchorEntity = AnchorEntity(world: position)
        anchorEntity.addChild(suggestionEntity)
        arView.scene.addAnchor(anchorEntity)
        
        contextualOverlays[recommendation.id] = anchorEntity
    }
    
    private func createSuggestionBubble(_ recommendation: AIRecommendation) -> Entity {
        let bubbleEntity = Entity()
        
        // Create suggestion icon
        let iconMesh = MeshResource.generateSphere(radius: 0.025)
        let iconMaterial = SimpleMaterial(color: .systemYellow.withAlphaComponent(0.8), isMetallic: false)
        
        let iconEntity = ModelEntity(mesh: iconMesh, materials: [iconMaterial])
        bubbleEntity.addChild(iconEntity)
        
        // Create text label
        let textMesh = MeshResource.generateText(
            recommendation.title,
            extrusionDepth: 0.001,
            font: .systemFont(ofSize: 0.02),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byTruncatingTail
        )
        
        let textMaterial = SimpleMaterial(color: .white, isMetallic: false)
        let textEntity = ModelEntity(mesh: textMesh, materials: [textMaterial])
        textEntity.position = [0, 0.05, 0]
        
        bubbleEntity.addChild(textEntity)
        
        return bubbleEntity
    }
    
    private func showClauseList() {
        // Show interactive clause list
        speakResponse("Here are all the clauses in your document. Tap on any clause to learn more about it.")
    }
    
    private func showPrecedentComparison() {
        // Show precedent comparison visualization
        speakResponse("I've compared your document with similar agreements. The highlighted areas show where your document differs from standard practices.")
    }
    
    private func showTranslationOverlay(_ translations: [Translation]) {
        // Create translation overlays for each clause
        for translation in translations {
            guard let clause = currentDocument?.clauses.first(where: { $0.id == translation.clauseId }),
                  let arView = arView else { continue }
            
            let position = calculateClausePosition(clause)
            let translationEntity = createTranslationOverlay(translation)
            translationEntity.position = position + [-0.1, 0.05, 0]
            
            let anchorEntity = AnchorEntity(world: position)
            anchorEntity.addChild(translationEntity)
            arView.scene.addAnchor(anchorEntity)
            
            contextualOverlays[translation.clauseId] = anchorEntity
        }
    }
    
    private func createTranslationOverlay(_ translation: Translation) -> Entity {
        let overlayEntity = Entity()
        
        // Create translation flag/icon
        let flagMesh = MeshResource.generateSphere(radius: 0.015)
        let flagMaterial = SimpleMaterial(color: .systemCyan.withAlphaComponent(0.8), isMetallic: false)
        
        let flagEntity = ModelEntity(mesh: flagMesh, materials: [flagMaterial])
        flagEntity.components.set(InputTargetComponent())
        
        overlayEntity.addChild(flagEntity)
        
        return overlayEntity
    }
    
    private func addToConversationHistory(_ message: ConversationMessage) {
        conversationHistory.append(message)
        
        // Keep only last 20 messages
        if conversationHistory.count > 20 {
            conversationHistory.removeFirst()
        }
    }
    
    private func clearVisualizations() {
        for (_, overlay) in contextualOverlays {
            overlay.removeFromParent()
        }
        contextualOverlays.removeAll()
        
        for (_, bubble) in helpBubbles {
            bubble.removeFromParent()
        }
        helpBubbles.removeAll()
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension AIAssistant: SFSpeechRecognizerDelegate {
    public func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if !available && isListening {
            stopListening()
        }
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension AIAssistant: AVSpeechSynthesizerDelegate {
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        updateAvatarState(.idle)
    }
    
    public func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        updateAvatarState(.speaking)
    }
}

// MARK: - Supporting Types

public enum AssistantMode {
    case contextual
    case conversational
    case analytical
    case educational
}

private enum AvatarState {
    case idle
    case listening
    case thinking
    case speaking
}

private enum IntentType {
    case explain
    case analyze
    case suggest
    case compare
    case search
    case general
}

private struct UserIntent {
    let type: IntentType
    let entities: [String]
    let confidence: Float
}

private struct AIResponse {
    let text: String
    let confidence: Float
    let actions: [AIAction]
}

private enum AIAction {
    case highlightClause(UUID)
    case showRiskOverlay
    case showSuggestions
    case showClauseList
    case showPrecedentComparison
}

public struct AISuggestion: Identifiable {
    public let id = UUID()
    public let text: String
    public let confidence: Float
    public let type: SuggestionType
}

public enum SuggestionType {
    case riskWarning
    case improvement
    case clarification
    case precedent
}

public struct AIRecommendation: Identifiable {
    public let id: UUID
    public let clauseId: UUID
    public let type: RecommendationType
    public let title: String
    public let description: String
    public let priority: Int
    public let suggestedAction: String
}

public enum RecommendationType {
    case riskMitigation
    case languageImprovement
    case standardCompliance
    case negotiationPoint
}

private enum ConversationMessage {
    case user(String)
    case assistant(String)
}

private struct UserPreferences {
    var preferredLanguage = "en"
    var verbosity: Float = 0.5
    var autoTranslate = false
    var voiceSpeed: Float = 0.5
}

private struct AIKnowledgeBase {
    var legalTerms: [String: String] = [:]
    var contractPatterns: [String] = []
    var riskPatterns: [String] = []
    
    mutating func loadLegalTerms() {
        legalTerms = [
            "arbitration": "A method of dispute resolution outside of court",
            "indemnification": "Protection from liability or loss",
            "force majeure": "Unforeseeable circumstances that prevent fulfillment of contract"
        ]
    }
    
    mutating func loadContractPatterns() {
        contractPatterns = [
            "termination clause",
            "liability limitation",
            "intellectual property rights",
            "confidentiality agreement"
        ]
    }
    
    mutating func loadRiskPatterns() {
        riskPatterns = [
            "unlimited liability",
            "automatic renewal",
            "broad indemnification",
            "exclusive remedies"
        ]
    }
}

private struct ClauseExplanation {
    let clauseId: UUID
    let summary: String
    let details: String
    let recommendations: [String]
    let relatedClauses: [UUID]
}

private struct PrecedentAnalysis {
    let summary: String
    let similarities: [String]
    let differences: [String]
    let recommendations: [String]
}

private struct Translation {
    let clauseId: UUID
    let originalText: String
    let translatedText: String
    let targetLanguage: String
    let confidence: Float
}

// MARK: - Custom Entity Classes

private class AssistantAvatarEntity: Entity, HasModel {
    var bodyEntity: ModelEntity?
    var faceEntity: ModelEntity?
    var thinkingBubble: Entity?
    
    override init() {
        super.init()
    }
    
    required init() {
        super.init()
    }
}

private class HelpBubbleEntity: Entity, HasModel {
    let clause: ClauseData
    
    init(clause: ClauseData) {
        self.clause = clause
        super.init()
    }
    
    required init() {
        fatalError("init() has not been implemented")
    }
}