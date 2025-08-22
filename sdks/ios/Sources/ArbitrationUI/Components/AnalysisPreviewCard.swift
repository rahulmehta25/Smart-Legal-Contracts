import SwiftUI
import ArbitrationSDK

/// Preview card showing summary of analysis results
@available(iOS 15.0, macOS 12.0, *)
public struct AnalysisPreviewCard: View {
    
    // MARK: - Properties
    let result: ArbitrationAnalysisResult
    let action: () -> Void
    
    // MARK: - Initialization
    public init(result: ArbitrationAnalysisResult, action: @escaping () -> Void) {
        self.result = result
        self.action = action
    }
    
    // MARK: - Body
    public var body: some View {
        Button(action: action) {
            HStack(spacing: 16) {
                // Status indicator
                VStack {
                    ZStack {
                        Circle()
                            .fill(result.riskLevel.color.opacity(0.2))
                            .frame(width: 50, height: 50)
                        
                        Image(systemName: result.hasArbitration ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(result.riskLevel.color)
                    }
                    
                    Text("\(Int(result.confidence * 100))%")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(result.riskLevel.color)
                }
                
                // Result details
                VStack(alignment: .leading, spacing: 4) {
                    Text(result.hasArbitration ? "Arbitration Detected" : "No Arbitration Found")
                        .font(.headline)
                        .foregroundColor(.primary)
                        .multilineTextAlignment(.leading)
                    
                    Text(result.riskLevel.description)
                        .font(.subheadline)
                        .foregroundColor(result.riskLevel.color)
                        .fontWeight(.medium)
                    
                    Text(formatAnalysisDate(result.metadata.analysisDate))
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    // Quick stats
                    HStack(spacing: 16) {
                        StatItem(
                            icon: "magnifyingglass",
                            value: "\(result.keywordMatches.count)",
                            label: "Keywords"
                        )
                        
                        StatItem(
                            icon: "clock",
                            value: String(format: "%.1fs", result.processingTime),
                            label: "Time"
                        )
                        
                        if !result.recommendations.isEmpty {
                            StatItem(
                                icon: "lightbulb",
                                value: "\(result.recommendations.count)",
                                label: "Tips"
                            )
                        }
                    }
                }
                
                Spacer()
                
                // Action indicator
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(result.riskLevel.color.opacity(0.3), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 2)
        }
        .buttonStyle(PlainButtonStyle())
    }
    
    // MARK: - Helper Methods
    
    private func formatAnalysisDate(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Supporting Views

private struct StatItem: View {
    let icon: String
    let value: String
    let label: String
    
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.caption2)
                .fontWeight(.medium)
                .foregroundColor(.primary)
            
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Preview
#if DEBUG
@available(iOS 15.0, macOS 12.0, *)
struct AnalysisPreviewCard_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            // High risk result
            AnalysisPreviewCard(
                result: ArbitrationAnalysisResult(
                    id: "sample-1",
                    hasArbitration: true,
                    confidence: 0.89,
                    keywordMatches: [
                        KeywordMatch(
                            keyword: "arbitration",
                            range: "sample".startIndex..<"sample".endIndex,
                            context: "binding arbitration clause",
                            confidence: 0.9
                        ),
                        KeywordMatch(
                            keyword: "binding",
                            range: "sample".startIndex..<"sample".endIndex,
                            context: "binding arbitration clause",
                            confidence: 0.8
                        )
                    ],
                    exclusionMatches: [],
                    riskLevel: .high,
                    recommendations: [
                        "Review arbitration terms carefully",
                        "Consider legal consultation"
                    ],
                    metadata: AnalysisMetadata(
                        textLength: 1500,
                        processingTime: 2.1,
                        sdkVersion: "1.0.0",
                        modelVersion: "1.0.0",
                        analysisDate: Date().addingTimeInterval(-3600) // 1 hour ago
                    )
                )
            ) {
                print("Tapped high risk result")
            }
            
            // No arbitration result
            AnalysisPreviewCard(
                result: ArbitrationAnalysisResult(
                    id: "sample-2",
                    hasArbitration: false,
                    confidence: 0.15,
                    keywordMatches: [],
                    exclusionMatches: [],
                    riskLevel: .minimal,
                    recommendations: [
                        "No arbitration clause detected"
                    ],
                    metadata: AnalysisMetadata(
                        textLength: 800,
                        processingTime: 1.3,
                        sdkVersion: "1.0.0",
                        modelVersion: "1.0.0",
                        analysisDate: Date().addingTimeInterval(-86400) // 1 day ago
                    )
                )
            ) {
                print("Tapped no arbitration result")
            }
            
            // Medium risk result
            AnalysisPreviewCard(
                result: ArbitrationAnalysisResult(
                    id: "sample-3",
                    hasArbitration: true,
                    confidence: 0.65,
                    keywordMatches: [
                        KeywordMatch(
                            keyword: "dispute resolution",
                            range: "sample".startIndex..<"sample".endIndex,
                            context: "alternative dispute resolution",
                            confidence: 0.7
                        )
                    ],
                    exclusionMatches: [],
                    riskLevel: .medium,
                    recommendations: [
                        "Moderate confidence detection",
                        "Manual review recommended"
                    ],
                    metadata: AnalysisMetadata(
                        textLength: 2200,
                        processingTime: 3.5,
                        sdkVersion: "1.0.0",
                        modelVersion: "1.0.0",
                        analysisDate: Date().addingTimeInterval(-300) // 5 minutes ago
                    )
                )
            ) {
                print("Tapped medium risk result")
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .previewLayout(.sizeThatFits)
    }
}
#endif