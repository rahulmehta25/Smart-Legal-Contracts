/**
 * documentGenerator.ts - Document Generation Utilities
 * 
 * Comprehensive document generation service with multiple format support,
 * template processing, and variable substitution.
 */

import {
  ContractDraft,
  ExportConfiguration,
  DocumentFormat,
  GeneratedSection,
  GeneratedClause,
  DocumentStyling,
  HeaderFooterConfig,
  WatermarkConfig
} from '../types';

interface DocumentProcessor {
  processTemplate: (content: string, variables: Record<string, any>) => string;
  generateTableOfContents: (sections: GeneratedSection[]) => string;
  applyNumbering: (sections: GeneratedSection[], style: string) => GeneratedSection[];
  formatDocument: (content: string, styling: DocumentStyling) => string;
}

interface ExportProcessor {
  exportToPDF: (content: string, config: ExportConfiguration) => Promise<Blob>;
  exportToHTML: (content: string, config: ExportConfiguration) => Promise<Blob>;
  exportToDOCX: (content: string, config: ExportConfiguration) => Promise<Blob>;
  exportToJSON: (draft: ContractDraft, config: ExportConfiguration) => Promise<Blob>;
}

// Variable substitution engine
class VariableProcessor {
  private static readonly VARIABLE_PATTERN = /\{\{([^}]+)\}\}/g;
  private static readonly CONDITIONAL_PATTERN = /\{\{#if\s+([^}]+)\}\}([\s\S]*?)\{\{\/if\}\}/g;
  private static readonly LOOP_PATTERN = /\{\{#each\s+([^}]+)\}\}([\s\S]*?)\{\{\/each\}\}/g;

  static processVariables(content: string, variables: Record<string, any>): string {
    // Replace simple variables
    let processed = content.replace(this.VARIABLE_PATTERN, (match, variableName) => {
      const trimmedName = variableName.trim();
      const value = this.getNestedValue(variables, trimmedName);
      return value !== undefined ? String(value) : match;
    });

    // Process conditional blocks
    processed = this.processConditionals(processed, variables);

    // Process loops
    processed = this.processLoops(processed, variables);

    return processed;
  }

  private static getNestedValue(obj: Record<string, any>, path: string): any {
    return path.split('.').reduce((current, key) => {
      return current && current[key] !== undefined ? current[key] : undefined;
    }, obj);
  }

  private static processConditionals(content: string, variables: Record<string, any>): string {
    return content.replace(this.CONDITIONAL_PATTERN, (match, condition, block) => {
      const shouldInclude = this.evaluateCondition(condition.trim(), variables);
      return shouldInclude ? block : '';
    });
  }

  private static processLoops(content: string, variables: Record<string, any>): string {
    return content.replace(this.LOOP_PATTERN, (match, arrayPath, template) => {
      const array = this.getNestedValue(variables, arrayPath.trim());
      if (!Array.isArray(array)) return '';

      return array.map((item, index) => {
        const loopVariables = { ...variables, item, index };
        return this.processVariables(template, loopVariables);
      }).join('');
    });
  }

  private static evaluateCondition(condition: string, variables: Record<string, any>): boolean {
    try {
      // Simple condition evaluation (in production, use a proper expression parser)
      const operators = ['===', '!==', '>=', '<=', '>', '<', '&&', '||'];
      
      for (const op of operators) {
        if (condition.includes(op)) {
          const [left, right] = condition.split(op).map(s => s.trim());
          const leftValue = this.getNestedValue(variables, left) ?? left;
          const rightValue = this.getNestedValue(variables, right) ?? right;

          switch (op) {
            case '===': return leftValue === rightValue;
            case '!==': return leftValue !== rightValue;
            case '>=': return Number(leftValue) >= Number(rightValue);
            case '<=': return Number(leftValue) <= Number(rightValue);
            case '>': return Number(leftValue) > Number(rightValue);
            case '<': return Number(leftValue) < Number(rightValue);
            case '&&': return Boolean(leftValue) && Boolean(rightValue);
            case '||': return Boolean(leftValue) || Boolean(rightValue);
          }
        }
      }

      // Simple boolean evaluation
      const value = this.getNestedValue(variables, condition);
      return Boolean(value);
    } catch (error) {
      console.warn('Error evaluating condition:', condition, error);
      return false;
    }
  }
}

// Document formatting utilities
class DocumentFormatter {
  static generateTitlePage(draft: ContractDraft, config: ExportConfiguration): string {
    const title = draft.variables['contract-title'] || draft.name;
    const parties = draft.parties.map(p => p.name).join(' and ');
    const effectiveDate = draft.variables['effective-date'] 
      ? new Date(draft.variables['effective-date']).toLocaleDateString()
      : 'To be determined';

    return `
      <div class="title-page">
        <div class="title-section">
          <h1 class="contract-title">${title}</h1>
          <div class="subtitle">Contract Agreement</div>
        </div>
        
        <div class="parties-section">
          <h2>Parties</h2>
          <p>${parties}</p>
        </div>
        
        <div class="metadata-section">
          <div class="metadata-item">
            <strong>Effective Date:</strong> ${effectiveDate}
          </div>
          <div class="metadata-item">
            <strong>Jurisdiction:</strong> ${draft.jurisdiction}
          </div>
          <div class="metadata-item">
            <strong>Version:</strong> ${draft.version}
          </div>
        </div>
        
        <div class="footer-section">
          <p>Generated on ${new Date().toLocaleDateString()}</p>
        </div>
      </div>
    `;
  }

  static generateTableOfContents(sections: GeneratedSection[]): string {
    const tocItems = sections.map(section => {
      const sectionToc = `<div class="toc-item level-1">
        <span class="toc-number">${section.numbering}</span>
        <span class="toc-title">${section.title}</span>
        <span class="toc-dots"></span>
        <span class="toc-page">${section.order}</span>
      </div>`;

      const clausesToc = section.clauses.map(clause => `
        <div class="toc-item level-2">
          <span class="toc-number">${clause.numbering}</span>
          <span class="toc-title">${clause.title}</span>
          <span class="toc-dots"></span>
          <span class="toc-page">${section.order}</span>
        </div>
      `).join('');

      return sectionToc + clausesToc;
    }).join('');

    return `
      <div class="table-of-contents">
        <h2>Table of Contents</h2>
        <div class="toc-content">
          ${tocItems}
        </div>
      </div>
    `;
  }

  static generateSignaturePage(draft: ContractDraft): string {
    const signatureBlocks = draft.parties.map(party => `
      <div class="signature-block">
        <h3>${party.role}</h3>
        <div class="party-info">
          <div class="party-name">${party.name}</div>
          <div class="party-address">
            ${party.contact.address.street}<br>
            ${party.contact.address.city}, ${party.contact.address.state} ${party.contact.address.postalCode}<br>
            ${party.contact.address.country}
          </div>
        </div>
        
        <div class="signature-section">
          <div class="signature-line">
            <div class="signature-placeholder"></div>
            <div class="signature-label">Signature</div>
          </div>
          
          <div class="signatory-info">
            <div class="signatory-name">${party.signatoryInfo.name}</div>
            <div class="signatory-title">${party.signatoryInfo.title}</div>
            <div class="signatory-date">Date: _________________</div>
          </div>
        </div>
      </div>
    `).join('');

    return `
      <div class="signature-page">
        <h2>Execution</h2>
        <p>The parties have executed this Agreement as of the date last signed below.</p>
        <div class="signature-blocks">
          ${signatureBlocks}
        </div>
      </div>
    `;
  }

  static applyWatermark(content: string, watermark: WatermarkConfig): string {
    const watermarkStyle = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%) rotate(${watermark.rotation}deg);
      font-size: ${watermark.fontSize}px;
      color: ${watermark.color};
      opacity: ${watermark.opacity};
      z-index: -1;
      pointer-events: none;
      user-select: none;
    `;

    const watermarkDiv = `
      <div class="watermark" style="${watermarkStyle}">
        ${watermark.text}
      </div>
    `;

    return content.replace('<body>', `<body>${watermarkDiv}`);
  }

  static applyHeaderFooter(content: string, config: HeaderFooterConfig): string {
    const headerContent = config.header ? `
      <div class="page-header">
        <div class="header-left">${config.header.left || ''}</div>
        <div class="header-center">${config.header.center || ''}</div>
        <div class="header-right">${config.header.right || ''}</div>
      </div>
    ` : '';

    const footerContent = config.footer ? `
      <div class="page-footer">
        <div class="footer-left">${config.footer.left || ''}</div>
        <div class="footer-center">${config.footer.center || ''}</div>
        <div class="footer-right">${config.footer.right || ''}</div>
        ${config.includePageNumbers ? '<div class="page-number"></div>' : ''}
        ${config.includeDate ? `<div class="date">${new Date().toLocaleDateString()}</div>` : ''}
      </div>
    ` : '';

    return headerContent + content + footerContent;
  }
}

// Main document generator class
export class DocumentGenerator implements DocumentProcessor {
  processTemplate(content: string, variables: Record<string, any>): string {
    return VariableProcessor.processVariables(content, variables);
  }

  generateTableOfContents(sections: GeneratedSection[]): string {
    return DocumentFormatter.generateTableOfContents(sections);
  }

  applyNumbering(sections: GeneratedSection[], style: string = 'numeric'): GeneratedSection[] {
    return sections.map((section, sectionIndex) => {
      const sectionNumber = this.formatSectionNumber(sectionIndex + 1, style);
      
      const numberedClauses = section.clauses.map((clause, clauseIndex) => {
        const clauseNumber = this.formatClauseNumber(sectionIndex + 1, clauseIndex + 1, style);
        return {
          ...clause,
          numbering: clauseNumber
        };
      });

      return {
        ...section,
        numbering: sectionNumber,
        clauses: numberedClauses
      };
    });
  }

  formatDocument(content: string, styling: DocumentStyling): string {
    const styles = `
      <style>
        body {
          font-family: ${styling.fontFamily};
          font-size: ${styling.fontSize}pt;
          line-height: ${styling.lineSpacing};
          margin: ${styling.margins.top}in ${styling.margins.right}in ${styling.margins.bottom}in ${styling.margins.left}in;
        }
        
        ${styling.headingStyles.map(heading => `
          h${heading.level} {
            font-family: ${heading.fontFamily || styling.fontFamily};
            font-size: ${heading.fontSize}pt;
            font-weight: ${heading.bold ? 'bold' : 'normal'};
            font-style: ${heading.italic ? 'italic' : 'normal'};
            color: ${heading.color || 'inherit'};
            margin-top: ${heading.spacing.before}pt;
            margin-bottom: ${heading.spacing.after}pt;
          }
        `).join('')}
        
        .page-break {
          page-break-after: always;
        }
        
        .signature-line {
          border-bottom: 1px solid #000;
          width: 200px;
          margin: 20px 0 5px 0;
        }
        
        .toc-item {
          display: flex;
          margin: 5px 0;
        }
        
        .toc-dots {
          flex: 1;
          border-bottom: 1px dotted #ccc;
          margin: 0 10px;
        }
        
        .watermark {
          position: fixed;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          opacity: 0.1;
          font-size: 72pt;
          color: #ccc;
          z-index: -1;
        }
      </style>
    `;

    return `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="utf-8">
          <title>Contract Document</title>
          ${styles}
        </head>
        <body>
          ${content}
        </body>
      </html>
    `;
  }

  private formatSectionNumber(index: number, style: string): string {
    switch (style.toLowerCase()) {
      case 'alphabetic':
        return String.fromCharCode(64 + index); // A, B, C...
      case 'roman':
        return this.toRoman(index);
      case 'decimal':
        return `${index}.0`;
      default:
        return String(index);
    }
  }

  private formatClauseNumber(sectionIndex: number, clauseIndex: number, style: string): string {
    const sectionNum = this.formatSectionNumber(sectionIndex, style);
    return `${sectionNum}.${clauseIndex}`;
  }

  private toRoman(num: number): string {
    const values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    const symbols = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I'];
    
    let result = '';
    for (let i = 0; i < values.length; i++) {
      while (num >= values[i]) {
        result += symbols[i];
        num -= values[i];
      }
    }
    return result;
  }
}

// Export processor implementation
export class ExportProcessor implements ExportProcessor {
  private documentGenerator: DocumentGenerator;

  constructor() {
    this.documentGenerator = new DocumentGenerator();
  }

  async exportToPDF(content: string, config: ExportConfiguration): Promise<Blob> {
    // In a real implementation, this would use a library like jsPDF or Puppeteer
    const htmlContent = this.documentGenerator.formatDocument(content, config.styling);
    
    // Apply watermark if specified
    const finalContent = config.watermark 
      ? DocumentFormatter.applyWatermark(htmlContent, config.watermark)
      : htmlContent;

    // Mock PDF generation - in reality, use proper PDF library
    return new Blob([finalContent], { type: 'application/pdf' });
  }

  async exportToHTML(content: string, config: ExportConfiguration): Promise<Blob> {
    const htmlContent = this.documentGenerator.formatDocument(content, config.styling);
    
    // Apply watermark and header/footer if specified
    let finalContent = config.watermark 
      ? DocumentFormatter.applyWatermark(htmlContent, config.watermark)
      : htmlContent;

    if (config.headerFooter) {
      finalContent = DocumentFormatter.applyHeaderFooter(finalContent, config.headerFooter);
    }

    return new Blob([finalContent], { type: 'text/html' });
  }

  async exportToDOCX(content: string, config: ExportConfiguration): Promise<Blob> {
    // In a real implementation, this would use a library like docx or officegen
    const processedContent = this.documentGenerator.processTemplate(content, {});
    
    // Mock DOCX generation
    return new Blob([processedContent], { 
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
  }

  async exportToJSON(draft: ContractDraft, config: ExportConfiguration): Promise<Blob> {
    const exportData = {
      contract: draft,
      metadata: {
        exportedAt: new Date().toISOString(),
        format: 'JSON',
        includeComments: config.includeComments,
        includeTrackChanges: config.includeTrackChanges,
        includeMetadata: config.includeMetadata
      }
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    return new Blob([jsonContent], { type: 'application/json' });
  }

  async generateCompleteDocument(draft: ContractDraft, config: ExportConfiguration): Promise<string> {
    const sections: string[] = [];

    // Add title page if requested
    if (config.sections.some(s => s.type === 'title_page' && s.include)) {
      sections.push(DocumentFormatter.generateTitlePage(draft, config));
      sections.push('<div class="page-break"></div>');
    }

    // Add table of contents if requested
    if (config.sections.some(s => s.type === 'toc' && s.include)) {
      sections.push(this.documentGenerator.generateTableOfContents(draft.structure));
      sections.push('<div class="page-break"></div>');
    }

    // Add contract body
    if (config.sections.some(s => s.type === 'contract_body' && s.include)) {
      const contractBody = this.generateContractBody(draft);
      sections.push(contractBody);
    }

    // Add signature page if requested
    if (config.sections.some(s => s.type === 'signature_page' && s.include)) {
      sections.push('<div class="page-break"></div>');
      sections.push(DocumentFormatter.generateSignaturePage(draft));
    }

    return sections.join('\n');
  }

  private generateContractBody(draft: ContractDraft): string {
    const numberedSections = this.documentGenerator.applyNumbering(draft.structure);
    
    return numberedSections.map(section => {
      const clausesContent = section.clauses.map(clause => `
        <div class="clause">
          <h4 class="clause-title">${clause.numbering} ${clause.title}</h4>
          <div class="clause-content">
            ${this.documentGenerator.processTemplate(clause.content, draft.variables)}
          </div>
        </div>
      `).join('');

      return `
        <div class="section">
          <h3 class="section-title">${section.numbering} ${section.title}</h3>
          <div class="section-content">
            ${clausesContent}
          </div>
        </div>
      `;
    }).join('');
  }
}

// Export the main service
export const documentGeneratorService = {
  generator: new DocumentGenerator(),
  exporter: new ExportProcessor(),
  
  async exportContract(draft: ContractDraft, config: ExportConfiguration): Promise<Blob> {
    const exporter = new ExportProcessor();
    const content = await exporter.generateCompleteDocument(draft, config);
    
    switch (config.format) {
      case 'PDF':
        return exporter.exportToPDF(content, config);
      case 'HTML':
        return exporter.exportToHTML(content, config);
      case 'DOCX':
        return exporter.exportToDOCX(content, config);
      case 'JSON':
        return exporter.exportToJSON(draft, config);
      default:
        return new Blob([content], { type: 'text/plain' });
    }
  }
};