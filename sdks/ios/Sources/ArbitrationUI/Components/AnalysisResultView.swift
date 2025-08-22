import SwiftUI
import ArbitrationSDK

/// Detailed analysis result view with recommendations and visualizations
@available(iOS 15.0, macOS 12.0, *)
public struct AnalysisResultView: View {
    
    // MARK: - Properties
    let result: ArbitrationAnalysisResult
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab = 0
    @State private var showingShareSheet = false
    @State private var exportedData: Data?
    
    // MARK: - Body
    public var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Header with main result
                    resultHeaderSection
                    
                    // Tab selection
                    tabSelectionSection
                    
                    // Tab content
                    TabView(selection: $selectedTab) {
                        // Overview tab
                        overviewTab
                            .tag(0)
                        
                        // Details tab
                        detailsTab
                            .tag(1)
                        
                        // Recommendations tab
                        recommendationsTab
                            .tag(2)
                        
                        // Metadata tab
                        metadataTab
                            .tag(3)
                    }
                    .tabViewStyle(PageTabViewStyle(indexDisplayMode: .never))
                    .frame(minHeight: 400)
                }
                .padding()
            }
            .navigationTitle("Analysis Result")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Menu {
                        Button(action: shareResult) {
                            Label("Share Report", systemImage: "square.and.arrow.up")
                        }
                        
                        Button(action: exportToPDF) {
                            Label("Export PDF", systemImage: "doc.pdf")
                        }
                        
                        Button(action: copyToClipboard) {
                            Label("Copy Summary", systemImage: "doc.on.clipboard")
                        }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .sheet(isPresented: $showingShareSheet) {
                if let data = exportedData {
                    ShareSheet(activityItems: [data])
                }
            }
        }
    }
    
    // MARK: - View Components
    
    private var resultHeaderSection: some View {
        VStack(spacing: 16) {
            // Main result indicator
            ZStack {
                Circle()
                    .fill(result.riskLevel.color.opacity(0.2))
                    .frame(width: 100, height: 100)
                
                Circle()
                    .stroke(result.riskLevel.color, lineWidth: 4)
                    .frame(width: 100, height: 100)
                
                VStack {
                    Image(systemName: result.hasArbitration ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .font(.title)
                        .foregroundColor(result.riskLevel.color)
                    
                    Text("\(Int(result.confidence * 100))%")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(result.riskLevel.color)
                }
            }
            
            // Result summary
            VStack(spacing: 8) {
                Text(result.hasArbitration ? "Arbitration Clause Detected" : "No Arbitration Clause Found")
                    .font(.title2)
                    .fontWeight(.bold)
                    .multilineTextAlignment(.center)
                
                Text("Confidence: \(Int(result.confidence * 100))%")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                RiskLevelBadge(riskLevel: result.riskLevel)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
    
    private var tabSelectionSection: some View {
        HStack(spacing: 0) {
            ForEach(TabType.allCases.indices, id: \.self) { index in
                let tab = TabType.allCases[index]
                
                Button(action: { selectedTab = index }) {
                    VStack(spacing: 4) {
                        Image(systemName: tab.icon)
                            .font(.caption)
                        
                        Text(tab.title)
                            .font(.caption2)
                            .fontWeight(.medium)
                    }
                    .foregroundColor(selectedTab == index ? .blue : .secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
            }
        }
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
    
    private var overviewTab: some View {
        VStack(spacing: 16) {
            // Confidence chart
            ConfidenceChart(confidence: result.confidence)
            
            // Quick stats
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                StatCard(
                    title: "Keywords Found",
                    value: "\(result.keywordMatches.count)",
                    icon: "magnifyingglass",
                    color: .blue
                )
                
                StatCard(
                    title: "Exclusions",
                    value: "\(result.exclusionMatches.count)",
                    icon: "xmark.circle",
                    color: .orange
                )
                
                StatCard(
                    title: "Processing Time",
                    value: String(format: "%.1fs", result.processingTime),
                    icon: "clock",
                    color: .green
                )
                
                StatCard(
                    title: "Text Length",
                    value: "\(result.metadata.textLength)",
                    icon: "doc.text",
                    color: .purple
                )
            }
            
            Spacer()
        }
        .padding()
    }
    
    private var detailsTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Keyword matches
            if !result.keywordMatches.isEmpty {
                SectionHeader(title: "Detected Keywords", icon: "magnifyingglass")
                
                ForEach(result.keywordMatches) { match in
                    KeywordMatchCard(match: match)
                }
            }
            
            // Exclusion matches
            if !result.exclusionMatches.isEmpty {
                SectionHeader(title: "Exclusion Phrases", icon: "xmark.circle")
                
                ForEach(result.exclusionMatches) { match in
                    KeywordMatchCard(match: match, isExclusion: true)
                }
            }
            
            if result.keywordMatches.isEmpty && result.exclusionMatches.isEmpty {
                EmptyStateCard(
                    icon: "magnifyingglass",
                    title: "No Specific Matches",
                    description: "Analysis was based on general content patterns"
                )
            }
            
            Spacer()
        }
        .padding()
    }
    
    private var recommendationsTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Recommendations", icon: "lightbulb")
            
            if !result.recommendations.isEmpty {
                ForEach(Array(result.recommendations.enumerated()), id: \.offset) { index, recommendation in
                    RecommendationCard(
                        recommendation: recommendation,
                        priority: index == 0 ? .high : .medium
                    )
                }
            } else {
                EmptyStateCard(
                    icon: "lightbulb",
                    title: "No Recommendations",
                    description: "The analysis is complete with no additional suggestions"
                )
            }
            
            Spacer()
        }
        .padding()
    }
    
    private var metadataTab: some View {
        VStack(alignment: .leading, spacing: 16) {
            SectionHeader(title: "Analysis Details", icon: "info.circle")
            
            VStack(spacing: 12) {
                MetadataRow(label: "Analysis ID", value: result.id)
                MetadataRow(label: "Date", value: formatDate(result.metadata.analysisDate))
                MetadataRow(label: "SDK Version", value: result.metadata.sdkVersion)
                MetadataRow(label: "Model Version", value: result.metadata.modelVersion)
                MetadataRow(label: "Text Length", value: "\(result.metadata.textLength) characters")
                MetadataRow(label: "Processing Time", value: String(format: "%.3f seconds", result.processingTime))
                MetadataRow(label: "Risk Level", value: result.riskLevel.description)
                MetadataRow(label: "Confidence Score", value: String(format: "%.1f%%", result.confidence * 100))
            }
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(8)
            
            Spacer()
        }
        .padding()
    }
    
    // MARK: - Actions
    
    private func shareResult() {
        let reportData = generateReportData()
        exportedData = reportData
        showingShareSheet = true
    }
    
    private func exportToPDF() {
        // Generate PDF report
        let pdfData = generatePDFReport()
        exportedData = pdfData
        showingShareSheet = true
    }
    
    private func copyToClipboard() {
        let summary = generateTextSummary()
        UIPasteboard.general.string = summary
    }
    
    // MARK: - Helper Methods
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private func generateReportData() -> Data {
        let report = generateTextSummary()
        return report.data(using: .utf8) ?? Data()
    }
    
    private func generateTextSummary() -> String {
        var summary = "ARBITRATION ANALYSIS REPORT\n"
        summary += "==========================\n\n"
        summary += "Result: \(result.hasArbitration ? "Arbitration clause detected" : "No arbitration clause found")\n"
        summary += "Confidence: \(Int(result.confidence * 100))%\n"
        summary += "Risk Level: \(result.riskLevel.description)\n"
        summary += "Analysis Date: \(formatDate(result.metadata.analysisDate))\n\n"
        
        if !result.keywordMatches.isEmpty {
            summary += "DETECTED KEYWORDS:\n"
            for match in result.keywordMatches {
                summary += "- \(match.keyword) (confidence: \(Int(match.confidence * 100))%)\n"
            }
            summary += "\n"
        }
        
        if !result.recommendations.isEmpty {
            summary += "RECOMMENDATIONS:\n"
            for (index, recommendation) in result.recommendations.enumerated() {
                summary += "\(index + 1). \(recommendation)\n"
            }
        }
        
        return summary
    }
    
    private func generatePDFReport() -> Data {
        // Simplified PDF generation - in a real implementation, you'd use PDFKit
        return generateReportData()
    }
}

// MARK: - Supporting Views

private enum TabType: CaseIterable {
    case overview, details, recommendations, metadata
    
    var title: String {
        switch self {
        case .overview: return "Overview"
        case .details: return "Details"
        case .recommendations: return "Tips"
        case .metadata: return "Info"
        }
    }
    
    var icon: String {
        switch self {
        case .overview: return "chart.bar"
        case .details: return "list.bullet"
        case .recommendations: return "lightbulb"
        case .metadata: return "info.circle"
        }
    }
}

private struct RiskLevelBadge: View {
    let riskLevel: RiskLevel
    
    var body: some View {
        Text(riskLevel.description)
            .font(.caption)
            .fontWeight(.semibold)
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(riskLevel.color.opacity(0.2))
            .foregroundColor(riskLevel.color)
            .cornerRadius(12)
    }
}

private struct ConfidenceChart: View {
    let confidence: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Confidence Level")
                .font(.headline)
            
            ZStack(alignment: .leading) {
                Rectangle()
                    .fill(Color(.systemGray5))
                    .frame(height: 8)
                    .cornerRadius(4)
                
                Rectangle()
                    .fill(confidenceColor)
                    .frame(width: CGFloat(confidence) * 200, height: 8)
                    .cornerRadius(4)
                    .animation(.easeInOut(duration: 1.0), value: confidence)
            }
            .frame(width: 200)
            
            Text("\(Int(confidence * 100))% confident")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
    
    private var confidenceColor: Color {
        switch confidence {
        case 0.8...:
            return .green
        case 0.6..<0.8:
            return .yellow
        case 0.4..<0.6:
            return .orange
        default:
            return .red
        }
    }
}

private struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.title3)
                .fontWeight(.bold)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

private struct SectionHeader: View {
    let title: String
    let icon: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
            
            Text(title)
                .font(.headline)
                .fontWeight(.semibold)
            
            Spacer()
        }
    }
}

private struct KeywordMatchCard: View {
    let match: KeywordMatch
    let isExclusion: Bool
    
    init(match: KeywordMatch, isExclusion: Bool = false) {
        self.match = match
        self.isExclusion = isExclusion
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(match.keyword)
                    .font(.subheadline)
                    .fontWeight(.medium)
                
                Spacer()
                
                Text("\(Int(match.confidence * 100))%")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(isExclusion ? Color.orange.opacity(0.2) : Color.blue.opacity(0.2))
                    .foregroundColor(isExclusion ? .orange : .blue)
                    .cornerRadius(4)
            }
            
            Text(match.context)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isExclusion ? Color.orange.opacity(0.3) : Color.blue.opacity(0.3), lineWidth: 1)
        )
    }
}

private struct RecommendationCard: View {
    let recommendation: String
    let priority: Priority
    
    enum Priority {
        case high, medium, low
        
        var color: Color {
            switch self {
            case .high: return .red
            case .medium: return .orange
            case .low: return .blue
            }
        }
        
        var icon: String {
            switch self {
            case .high: return "exclamationmark.triangle.fill"
            case .medium: return "info.circle.fill"
            case .low: return "lightbulb.fill"
            }
        }
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: priority.icon)
                .foregroundColor(priority.color)
                .font(.title3)
            
            Text(recommendation)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
            
            Spacer()
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(8)
    }
}

private struct MetadataRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .fontWeight(.medium)
        }
        .font(.subheadline)
    }
}

private struct EmptyStateCard: View {
    let icon: String
    let title: String
    let description: String
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.largeTitle)
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

// MARK: - Share Sheet
private struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// MARK: - Preview
#if DEBUG
@available(iOS 15.0, macOS 12.0, *)
struct AnalysisResultView_Previews: PreviewProvider {
    static var previews: some View {
        let sampleResult = ArbitrationAnalysisResult(
            id: "sample-123",
            hasArbitration: true,
            confidence: 0.85,
            keywordMatches: [
                KeywordMatch(
                    keyword: "arbitration",
                    range: "sample text".startIndex..<"sample text".endIndex,
                    context: "...disputes shall be resolved through binding arbitration...",
                    confidence: 0.9
                )
            ],
            exclusionMatches: [],
            riskLevel: .high,
            recommendations: [
                "This document contains a binding arbitration clause that may limit your legal options.",
                "Consider consulting with a legal professional for detailed analysis."
            ],
            metadata: AnalysisMetadata(
                textLength: 1500,
                processingTime: 2.3,
                sdkVersion: "1.0.0",
                modelVersion: "1.0.0",
                analysisDate: Date()
            )
        )
        
        AnalysisResultView(result: sampleResult)
    }
}
#endif