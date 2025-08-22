import SwiftUI
import ArbitrationSDK
import PhotosUI
import UniformTypeIdentifiers

/// Main document analyzer view with text input and file upload capabilities
@available(iOS 15.0, macOS 12.0, *)
public struct DocumentAnalyzerView: View {
    
    // MARK: - Properties
    @StateObject private var detector = ArbitrationDetector()
    @State private var inputText = ""
    @State private var analysisResult: ArbitrationAnalysisResult?
    @State private var showingResult = false
    @State private var showingImagePicker = false
    @State private var showingDocumentPicker = false
    @State private var errorMessage: String?
    @State private var showingError = false
    
    // Document picker
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    
    // Configuration
    public let configuration: AnalyzerConfiguration
    
    // MARK: - Initialization
    public init(configuration: AnalyzerConfiguration = .default) {
        self.configuration = configuration
    }
    
    // MARK: - Body
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header
                    headerSection
                    
                    // Input methods
                    inputMethodsSection
                    
                    // Text input
                    textInputSection
                    
                    // Action buttons
                    actionButtonsSection
                    
                    // Progress indicator
                    if detector.isAnalyzing {
                        progressSection
                    }
                    
                    // Recent results
                    if !detector.isAnalyzing {
                        recentResultsSection
                    }
                }
                .padding()
            }
            .navigationTitle("Arbitration Analyzer")
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showingResult) {
                if let result = analysisResult {
                    AnalysisResultView(result: result)
                }
            }
            .photosPicker(
                isPresented: $showingImagePicker,
                selection: $selectedItem,
                matching: .images,
                photoLibrary: .shared()
            )
            .fileImporter(
                isPresented: $showingDocumentPicker,
                allowedContentTypes: [.pdf, .text, .rtf],
                allowsMultipleSelection: false
            ) { result in
                handleDocumentSelection(result)
            }
            .alert("Error", isPresented: $showingError) {
                Button("OK") { errorMessage = nil }
            } message: {
                Text(errorMessage ?? "An unknown error occurred")
            }
            .onChange(of: selectedItem) { item in
                Task {
                    await loadSelectedImage(item)
                }
            }
        }
    }
    
    // MARK: - View Components
    
    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 48))
                .foregroundColor(.blue)
            
            Text("Analyze Legal Documents")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Detect arbitration clauses and assess legal risks")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.vertical)
    }
    
    private var inputMethodsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Choose Input Method")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                InputMethodCard(
                    icon: "keyboard",
                    title: "Type Text",
                    description: "Enter text directly",
                    action: { /* Already active */ }
                )
                
                InputMethodCard(
                    icon: "camera",
                    title: "Scan Image",
                    description: "OCR from photo",
                    action: { showingImagePicker = true }
                )
                
                InputMethodCard(
                    icon: "doc",
                    title: "Upload File",
                    description: "PDF, Word, Text",
                    action: { showingDocumentPicker = true }
                )
                
                InputMethodCard(
                    icon: "mic",
                    title: "Voice Input",
                    description: "Coming soon",
                    action: { },
                    isDisabled: true
                )
            }
        }
    }
    
    private var textInputSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Document Text")
                    .font(.headline)
                
                Spacer()
                
                Text("\(inputText.count) characters")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            TextEditor(text: $inputText)
                .frame(minHeight: 150)
                .padding(8)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 1)
                )
                .overlay(
                    Group {
                        if inputText.isEmpty {
                            VStack {
                                HStack {
                                    Text("Paste or type your legal document text here...")
                                        .foregroundColor(.secondary)
                                        .padding(.leading, 12)
                                        .padding(.top, 16)
                                    Spacer()
                                }
                                Spacer()
                            }
                        }
                    }
                )
        }
    }
    
    private var actionButtonsSection: some View {
        VStack(spacing: 12) {
            Button(action: analyzeText) {
                HStack {
                    if detector.isAnalyzing {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "magnifyingglass")
                    }
                    
                    Text(detector.isAnalyzing ? "Analyzing..." : "Analyze Document")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(inputText.isEmpty ? Color.gray : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .disabled(inputText.isEmpty || detector.isAnalyzing)
            
            if detector.isAnalyzing {
                Button("Cancel Analysis") {
                    detector.cancelAnalysis()
                }
                .foregroundColor(.red)
            }
        }
    }
    
    private var progressSection: some View {
        VStack(spacing: 12) {
            ProgressView(value: detector.analysisProgress)
                .progressViewStyle(LinearProgressViewStyle())
            
            Text("Analyzing document...")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Text("\(Int(detector.analysisProgress * 100))% complete")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(10)
    }
    
    private var recentResultsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Recent Analysis")
                    .font(.headline)
                
                Spacer()
                
                if let result = detector.lastAnalysisResult {
                    Button("View Details") {
                        analysisResult = result
                        showingResult = true
                    }
                    .font(.caption)
                }
            }
            
            if let result = detector.lastAnalysisResult {
                AnalysisPreviewCard(result: result) {
                    analysisResult = result
                    showingResult = true
                }
            } else {
                EmptyStateView(
                    icon: "doc.text.magnifyingglass",
                    title: "No Analysis Yet",
                    description: "Analyze a document to see results here"
                )
            }
        }
    }
    
    // MARK: - Actions
    
    private func analyzeText() {
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        
        Task {
            do {
                let result = try await detector.analyzeText(inputText)
                await MainActor.run {
                    analysisResult = result
                    showingResult = true
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    showingError = true
                }
            }
        }
    }
    
    private func loadSelectedImage(_ item: PhotosPickerItem?) async {
        guard let item = item else { return }
        
        do {
            if let data = try await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                await MainActor.run {
                    selectedImage = image
                }
                
                // Analyze image with OCR
                let result = try await detector.analyzeImage(image)
                await MainActor.run {
                    analysisResult = result
                    showingResult = true
                }
            }
        } catch {
            await MainActor.run {
                errorMessage = "Failed to process image: \(error.localizedDescription)"
                showingError = true
            }
        }
    }
    
    private func handleDocumentSelection(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            
            Task {
                do {
                    let data = try Data(contentsOf: url)
                    let documentType: DocumentType
                    
                    switch url.pathExtension.lowercased() {
                    case "pdf":
                        documentType = .pdf
                    case "txt":
                        documentType = .text
                    case "rtf":
                        documentType = .rtf
                    default:
                        documentType = .text
                    }
                    
                    let analysisResult = try await detector.analyzeDocument(data, type: documentType)
                    
                    await MainActor.run {
                        self.analysisResult = analysisResult
                        showingResult = true
                    }
                } catch {
                    await MainActor.run {
                        errorMessage = "Failed to analyze document: \(error.localizedDescription)"
                        showingError = true
                    }
                }
            }
            
        case .failure(let error):
            errorMessage = "Failed to select document: \(error.localizedDescription)"
            showingError = true
        }
    }
}

// MARK: - Supporting Views

private struct InputMethodCard: View {
    let icon: String
    let title: String
    let description: String
    let action: () -> Void
    let isDisabled: Bool
    
    init(icon: String, title: String, description: String, action: @escaping () -> Void, isDisabled: Bool = false) {
        self.icon = icon
        self.title = title
        self.description = description
        self.action = action
        self.isDisabled = isDisabled
    }
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(isDisabled ? .gray : .blue)
                
                VStack(spacing: 2) {
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(isDisabled ? .gray : .primary)
                    
                    Text(description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(8)
        }
        .disabled(isDisabled)
    }
}

private struct EmptyStateView: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 40))
                .foregroundColor(.gray)
            
            Text(title)
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text(description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

// MARK: - Configuration

public struct AnalyzerConfiguration {
    public let showRecentResults: Bool
    public let allowFileUpload: Bool
    public let allowImageScanning: Bool
    public let maxTextLength: Int
    
    public init(
        showRecentResults: Bool = true,
        allowFileUpload: Bool = true,
        allowImageScanning: Bool = true,
        maxTextLength: Int = 50000
    ) {
        self.showRecentResults = showRecentResults
        self.allowFileUpload = allowFileUpload
        self.allowImageScanning = allowImageScanning
        self.maxTextLength = maxTextLength
    }
    
    public static let `default` = AnalyzerConfiguration()
}

// MARK: - Preview
#if DEBUG
@available(iOS 15.0, macOS 12.0, *)
struct DocumentAnalyzerView_Previews: PreviewProvider {
    static var previews: some View {
        DocumentAnalyzerView()
    }
}
#endif