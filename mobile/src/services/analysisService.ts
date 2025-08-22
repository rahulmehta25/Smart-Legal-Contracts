import { ArbitrationAnalysis, Document, ArbitrationClause, RiskLevel, ClauseType, SeverityLevel } from '@types/index';
import { offlineService } from './offline';
import { documentService } from './documentService';
import { generateUUID } from '@utils/helpers';

class AnalysisService {
  private readonly API_ENDPOINT = 'https://api.arbitrationdetector.com'; // Replace with actual endpoint

  async analyzeDocument(document: Document): Promise<ArbitrationAnalysis> {
    try {
      // Update document status to processing
      await documentService.updateAnalysisStatus(document.id, 'processing');

      let analysis: ArbitrationAnalysis;

      try {
        // Try online analysis first
        analysis = await this.performOnlineAnalysis(document);
      } catch (error) {
        console.log('Online analysis failed, falling back to offline:', error);
        // Fallback to offline analysis
        analysis = await this.performOfflineAnalysis(document);
      }

      // Save analysis result
      await offlineService.saveAnalysis(analysis);
      
      // Update document status to completed
      await documentService.updateAnalysisStatus(document.id, 'completed');

      return analysis;
    } catch (error) {
      console.error('Error analyzing document:', error);
      
      // Update document status to failed
      await documentService.updateAnalysisStatus(document.id, 'failed');
      
      throw new Error('Failed to analyze document');
    }
  }

  async getAnalysisByDocumentId(documentId: string): Promise<ArbitrationAnalysis | null> {
    try {
      return await offlineService.getAnalysis(documentId);
    } catch (error) {
      console.error('Error getting analysis:', error);
      throw new Error('Failed to retrieve analysis');
    }
  }

  async getAllAnalyses(): Promise<ArbitrationAnalysis[]> {
    try {
      return await offlineService.getAllAnalyses();
    } catch (error) {
      console.error('Error getting all analyses:', error);
      throw new Error('Failed to retrieve analyses');
    }
  }

  async deleteAnalysisByDocumentId(documentId: string): Promise<void> {
    try {
      // Implementation would delete from offline storage
      // For now, this is a placeholder
      console.log('Deleting analysis for document:', documentId);
    } catch (error) {
      console.error('Error deleting analysis:', error);
      throw new Error('Failed to delete analysis');
    }
  }

  async reanalyzeDocument(documentId: string): Promise<ArbitrationAnalysis> {
    try {
      const document = await documentService.getDocumentById(documentId);
      if (!document) {
        throw new Error('Document not found');
      }

      return await this.analyzeDocument(document);
    } catch (error) {
      console.error('Error reanalyzing document:', error);
      throw new Error('Failed to reanalyze document');
    }
  }

  private async performOnlineAnalysis(document: Document): Promise<ArbitrationAnalysis> {
    const startTime = Date.now();

    try {
      const response = await fetch(`${this.API_ENDPOINT}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer YOUR_API_TOKEN', // Replace with actual token
        },
        body: JSON.stringify({
          documentId: document.id,
          text: document.extractedText,
          metadata: {
            name: document.name,
            type: document.type,
            size: document.size,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const result = await response.json();
      const processingTime = Date.now() - startTime;

      return {
        id: generateUUID(),
        documentId: document.id,
        hasArbitrationClause: result.hasArbitrationClause,
        confidence: result.confidence,
        detectedClauses: result.detectedClauses.map((clause: any) => ({
          id: generateUUID(),
          text: clause.text,
          type: clause.type,
          startPosition: clause.startPosition,
          endPosition: clause.endPosition,
          confidence: clause.confidence,
          severity: clause.severity,
          explanation: clause.explanation,
        })),
        riskLevel: result.riskLevel,
        recommendations: result.recommendations,
        createdAt: new Date(),
        processingTime,
      };
    } catch (error) {
      console.error('Online analysis failed:', error);
      throw error;
    }
  }

  private async performOfflineAnalysis(document: Document): Promise<ArbitrationAnalysis> {
    const startTime = Date.now();

    try {
      // Perform local analysis using predefined patterns and rules
      const analysisResult = this.analyzeTextLocally(document.extractedText);
      const processingTime = Date.now() - startTime;

      return {
        id: generateUUID(),
        documentId: document.id,
        hasArbitrationClause: analysisResult.hasArbitrationClause,
        confidence: analysisResult.confidence,
        detectedClauses: analysisResult.detectedClauses,
        riskLevel: analysisResult.riskLevel,
        recommendations: analysisResult.recommendations,
        createdAt: new Date(),
        processingTime,
      };
    } catch (error) {
      console.error('Offline analysis failed:', error);
      throw error;
    }
  }

  private analyzeTextLocally(text: string): {
    hasArbitrationClause: boolean;
    confidence: number;
    detectedClauses: ArbitrationClause[];
    riskLevel: RiskLevel;
    recommendations: string[];
  } {
    const detectedClauses: ArbitrationClause[] = [];
    const lowercaseText = text.toLowerCase();

    // Define arbitration-related patterns
    const arbitrationPatterns = [
      {
        pattern: /arbitration|arbitrator|arbitral/gi,
        type: ClauseType.MANDATORY_ARBITRATION,
        severity: SeverityLevel.WARNING,
        weight: 0.8,
      },
      {
        pattern: /binding arbitration|mandatory arbitration/gi,
        type: ClauseType.MANDATORY_ARBITRATION,
        severity: SeverityLevel.CRITICAL,
        weight: 0.9,
      },
      {
        pattern: /class action waiver|no class action|waive.*class/gi,
        type: ClauseType.CLASS_ACTION_WAIVER,
        severity: SeverityLevel.CRITICAL,
        weight: 0.85,
      },
      {
        pattern: /dispute resolution|disputes.*resolved/gi,
        type: ClauseType.DISPUTE_RESOLUTION,
        severity: SeverityLevel.INFO,
        weight: 0.6,
      },
      {
        pattern: /jurisdiction|governing law|venue/gi,
        type: ClauseType.JURISDICTION_CLAUSE,
        severity: SeverityLevel.WARNING,
        weight: 0.5,
      },
    ];

    let totalConfidence = 0;
    let maxWeight = 0;

    // Search for patterns
    arbitrationPatterns.forEach(({ pattern, type, severity, weight }) => {
      const matches = text.match(pattern);
      if (matches) {
        matches.forEach((match, index) => {
          const startPosition = text.indexOf(match);
          const endPosition = startPosition + match.length;

          // Extract surrounding context
          const contextStart = Math.max(0, startPosition - 100);
          const contextEnd = Math.min(text.length, endPosition + 100);
          const context = text.substring(contextStart, contextEnd);

          const clause: ArbitrationClause = {
            id: generateUUID(),
            text: context,
            type,
            startPosition,
            endPosition,
            confidence: weight,
            severity,
            explanation: this.getClauseExplanation(type),
          };

          detectedClauses.push(clause);
        });

        totalConfidence += weight;
        maxWeight = Math.max(maxWeight, weight);
      }
    });

    // Calculate overall confidence and risk level
    const hasArbitrationClause = detectedClauses.length > 0;
    const confidence = hasArbitrationClause ? Math.min(maxWeight, 1.0) : 0;
    
    let riskLevel: RiskLevel;
    if (maxWeight >= 0.9) {
      riskLevel = RiskLevel.CRITICAL;
    } else if (maxWeight >= 0.8) {
      riskLevel = RiskLevel.HIGH;
    } else if (maxWeight >= 0.6) {
      riskLevel = RiskLevel.MEDIUM;
    } else {
      riskLevel = RiskLevel.LOW;
    }

    // Generate recommendations
    const recommendations = this.generateRecommendations(detectedClauses, riskLevel);

    return {
      hasArbitrationClause,
      confidence,
      detectedClauses,
      riskLevel,
      recommendations,
    };
  }

  private getClauseExplanation(type: ClauseType): string {
    switch (type) {
      case ClauseType.MANDATORY_ARBITRATION:
        return 'This clause requires disputes to be resolved through arbitration instead of court proceedings, which may limit your legal options.';
      case ClauseType.CLASS_ACTION_WAIVER:
        return 'This clause prevents you from joining class action lawsuits, forcing you to pursue individual legal action.';
      case ClauseType.DISPUTE_RESOLUTION:
        return 'This clause outlines how disputes will be handled, which may include mandatory arbitration or mediation.';
      case ClauseType.JURISDICTION_CLAUSE:
        return 'This clause specifies which courts or jurisdictions will handle disputes, potentially making legal action more difficult.';
      case ClauseType.VOLUNTARY_ARBITRATION:
        return 'This clause offers arbitration as an option for dispute resolution but does not make it mandatory.';
      default:
        return 'This clause may affect how disputes are resolved.';
    }
  }

  private generateRecommendations(clauses: ArbitrationClause[], riskLevel: RiskLevel): string[] {
    const recommendations: string[] = [];

    if (clauses.length === 0) {
      recommendations.push('No arbitration clauses detected. You should have standard legal recourse options.');
      return recommendations;
    }

    // General recommendations based on risk level
    switch (riskLevel) {
      case RiskLevel.CRITICAL:
        recommendations.push('⚠️ High-risk arbitration clauses detected. Consider consulting with a lawyer before signing.');
        recommendations.push('These clauses may significantly limit your legal rights and options for dispute resolution.');
        break;
      case RiskLevel.HIGH:
        recommendations.push('⚠️ Significant arbitration clauses found. Review carefully before agreeing.');
        recommendations.push('Consider negotiating these terms or seeking legal advice.');
        break;
      case RiskLevel.MEDIUM:
        recommendations.push('Some arbitration-related clauses detected. Review the implications carefully.');
        break;
      case RiskLevel.LOW:
        recommendations.push('Minor arbitration-related language found. Generally low risk.');
        break;
    }

    // Specific recommendations based on clause types
    const clauseTypes = new Set(clauses.map(c => c.type));

    if (clauseTypes.has(ClauseType.MANDATORY_ARBITRATION)) {
      recommendations.push('• Mandatory arbitration detected: This prevents you from taking disputes to court.');
    }

    if (clauseTypes.has(ClauseType.CLASS_ACTION_WAIVER)) {
      recommendations.push('• Class action waiver found: You cannot join group lawsuits against this entity.');
    }

    if (clauseTypes.has(ClauseType.JURISDICTION_CLAUSE)) {
      recommendations.push('• Jurisdiction restrictions: Disputes must be resolved in specific courts or locations.');
    }

    // General advice
    recommendations.push('📚 Research arbitration vs. court proceedings to understand the differences.');
    recommendations.push('🔍 Look for alternative service providers if these terms are unacceptable.');

    return recommendations;
  }

  async getAnalysisStatistics(): Promise<{
    totalAnalyses: number;
    documentsWithArbitration: number;
    averageConfidence: number;
    riskDistribution: Record<RiskLevel, number>;
  }> {
    try {
      const analyses = await this.getAllAnalyses();
      
      const riskDistribution = {
        [RiskLevel.LOW]: 0,
        [RiskLevel.MEDIUM]: 0,
        [RiskLevel.HIGH]: 0,
        [RiskLevel.CRITICAL]: 0,
      };

      let totalConfidence = 0;
      let documentsWithArbitration = 0;

      analyses.forEach(analysis => {
        if (analysis.hasArbitrationClause) {
          documentsWithArbitration++;
        }
        totalConfidence += analysis.confidence;
        riskDistribution[analysis.riskLevel]++;
      });

      return {
        totalAnalyses: analyses.length,
        documentsWithArbitration,
        averageConfidence: analyses.length > 0 ? totalConfidence / analyses.length : 0,
        riskDistribution,
      };
    } catch (error) {
      console.error('Error getting analysis statistics:', error);
      throw new Error('Failed to get analysis statistics');
    }
  }
}

export const analysisService = new AnalysisService();