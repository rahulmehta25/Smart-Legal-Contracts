import { IAccessibilityAuditResult, IAccessibilityIssue } from '@/types/accessibility';

export class AccessibilityAuditor {
  public async audit(): Promise<IAccessibilityAuditResult> {
    const issues: IAccessibilityIssue[] = [];
    
    // Perform basic accessibility checks
    this.checkImages(issues);
    this.checkHeadings(issues);
    this.checkLinks(issues);
    this.checkFormLabels(issues);

    const score = this.calculateScore(issues);
    
    return {
      score,
      issues,
      recommendations: this.generateRecommendations(issues),
      timestamp: new Date()
    };
  }

  private checkImages(issues: IAccessibilityIssue[]): void {
    const images = document.querySelectorAll('img:not([alt])');
    images.forEach((img, index) => {
      issues.push({
        type: 'error',
        element: `img:nth-child(${index + 1})`,
        description: 'Image missing alt attribute',
        wcagLevel: 'A',
        impact: 'high',
        fix: 'Add descriptive alt text to the image'
      });
    });
  }

  private checkHeadings(issues: IAccessibilityIssue[]): void {
    // Basic heading structure check
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (headings.length === 0) {
      issues.push({
        type: 'warning',
        element: 'document',
        description: 'No heading elements found',
        wcagLevel: 'AA',
        impact: 'medium',
        fix: 'Add proper heading structure to the page'
      });
    }
  }

  private checkLinks(issues: IAccessibilityIssue[]): void {
    const links = document.querySelectorAll('a:not([href])');
    links.forEach((link, index) => {
      issues.push({
        type: 'error',
        element: `a:nth-child(${index + 1})`,
        description: 'Link missing href attribute',
        wcagLevel: 'A',
        impact: 'high',
        fix: 'Add a valid href attribute to the link'
      });
    });
  }

  private checkFormLabels(issues: IAccessibilityIssue[]): void {
    const inputs = document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])');
    inputs.forEach((input, index) => {
      const hasLabel = document.querySelector(`label[for="${input.id}"]`);
      if (!hasLabel && input.id) {
        issues.push({
          type: 'error',
          element: `input#${input.id}`,
          description: 'Form input missing label',
          wcagLevel: 'A',
          impact: 'high',
          fix: 'Add a label element or aria-label attribute'
        });
      }
    });
  }

  private calculateScore(issues: IAccessibilityIssue[]): number {
    let score = 100;
    issues.forEach(issue => {
      switch (issue.impact) {
        case 'critical': score -= 20; break;
        case 'high': score -= 10; break;
        case 'medium': score -= 5; break;
        case 'low': score -= 2; break;
      }
    });
    return Math.max(0, score);
  }

  private generateRecommendations(issues: IAccessibilityIssue[]): string[] {
    const recommendations: string[] = [];
    
    if (issues.some(i => i.type === 'error')) {
      recommendations.push('Fix all critical accessibility errors');
    }
    
    if (issues.some(i => i.description.includes('alt'))) {
      recommendations.push('Ensure all images have descriptive alt text');
    }
    
    if (issues.some(i => i.description.includes('heading'))) {
      recommendations.push('Implement proper heading hierarchy');
    }
    
    return recommendations;
  }
}