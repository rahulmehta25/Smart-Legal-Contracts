import { Page, expect } from '@playwright/test';
import path from 'path';

export interface UploadOptions {
  waitForAnalysis?: boolean;
  expectedResult?: 'arbitration' | 'no-arbitration' | 'uncertain';
  timeout?: number;
}

/**
 * Upload a document and optionally wait for analysis results
 */
export async function uploadDocument(
  page: Page, 
  filename: string, 
  options: UploadOptions = {}
) {
  const {
    waitForAnalysis = true,
    expectedResult,
    timeout = 30000
  } = options;

  console.log(`📤 Uploading document: ${filename}`);
  
  // Navigate to upload page if not already there
  const currentUrl = page.url();
  if (!currentUrl.includes('/upload') && !currentUrl.includes('/')) {
    await page.goto('/upload');
  }

  // Wait for upload component to be ready
  await page.waitForSelector('[data-testid="document-uploader"]', { timeout });
  
  // Handle file upload
  const fileInput = page.locator('[data-testid="file-input"]');
  const filePath = path.join(__dirname, '../../sample-data/documents', filename);
  
  await fileInput.setInputFiles(filePath);
  
  // Verify file is selected
  await expect(page.locator('[data-testid="selected-file"]')).toContainText(filename);
  
  // Click upload button
  await page.click('[data-testid="upload-button"]');
  
  if (waitForAnalysis) {
    console.log('⏳ Waiting for analysis to complete...');
    
    // Wait for upload progress
    await page.waitForSelector('[data-testid="upload-progress"]', { timeout });
    
    // Wait for analysis to start
    await page.waitForSelector('[data-testid="analysis-status"]', { timeout });
    
    // Wait for analysis to complete
    await page.waitForSelector('[data-testid="analysis-complete"]', { 
      timeout: timeout * 2 // Analysis might take longer
    });
    
    // Verify expected result if provided
    if (expectedResult) {
      const resultElement = page.locator('[data-testid="analysis-result"]');
      await expect(resultElement).toBeVisible();
      
      switch (expectedResult) {
        case 'arbitration':
          await expect(resultElement).toContainText('Arbitration clause detected');
          await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
          break;
        case 'no-arbitration':
          await expect(resultElement).toContainText('No arbitration clause found');
          break;
        case 'uncertain':
          await expect(resultElement).toContainText('Uncertain');
          break;
      }
    }
    
    console.log('✅ Analysis completed successfully');
  }
  
  return {
    filename,
    uploaded: true,
    analysisCompleted: waitForAnalysis
  };
}

/**
 * Upload multiple documents in sequence
 */
export async function uploadMultipleDocuments(
  page: Page,
  filenames: string[],
  options: UploadOptions = {}
) {
  const results = [];
  
  for (const filename of filenames) {
    const result = await uploadDocument(page, filename, options);
    results.push(result);
    
    // Small delay between uploads
    await page.waitForTimeout(1000);
  }
  
  return results;
}

/**
 * Upload document via drag and drop
 */
export async function uploadDocumentViaDragDrop(
  page: Page,
  filename: string,
  options: UploadOptions = {}
) {
  console.log(`🎯 Uploading document via drag & drop: ${filename}`);
  
  const filePath = path.join(__dirname, '../../sample-data/documents', filename);
  
  // Wait for drop zone
  const dropZone = page.locator('[data-testid="drop-zone"]');
  await expect(dropZone).toBeVisible();
  
  // Create file input for drag and drop simulation
  const fileChooserPromise = page.waitForEvent('filechooser');
  await dropZone.click();
  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles(filePath);
  
  // Continue with standard upload flow
  if (options.waitForAnalysis) {
    await page.waitForSelector('[data-testid="analysis-complete"]', { 
      timeout: options.timeout || 30000 
    });
  }
  
  return { filename, uploaded: true, method: 'drag-drop' };
}

/**
 * Create test document data for uploads
 */
export async function createTestDocument(
  filename: string,
  content: string,
  type: 'arbitration' | 'no-arbitration' | 'mixed' = 'arbitration'
) {
  const templates = {
    arbitration: `
      TERMS OF SERVICE
      
      ${content}
      
      DISPUTE RESOLUTION
      Any dispute arising out of or relating to this agreement shall be resolved through binding arbitration 
      administered by the American Arbitration Association in accordance with its Commercial Arbitration Rules.
      The arbitration shall take place in New York, NY, and the decision of the arbitrator shall be final and binding.
      
      You agree to waive your right to participate in a class action lawsuit or class-wide arbitration.
    `,
    'no-arbitration': `
      TERMS OF SERVICE
      
      ${content}
      
      DISPUTE RESOLUTION
      Any disputes arising from this agreement shall be resolved through negotiation between the parties.
      If negotiation fails, disputes may be brought before the competent courts in the jurisdiction where
      the service provider is located.
    `,
    mixed: `
      TERMS OF SERVICE
      
      ${content}
      
      DISPUTE RESOLUTION
      For disputes under $5,000, the parties may choose between arbitration or small claims court.
      For larger disputes, arbitration may be required at the discretion of the service provider.
      Some disputes may be excluded from arbitration requirements.
    `
  };
  
  return {
    filename,
    content: templates[type],
    expectedResult: type === 'arbitration' ? 'arbitration' : 
                   type === 'no-arbitration' ? 'no-arbitration' : 'uncertain'
  };
}

/**
 * Upload test document for global setup
 */
export async function uploadTestDocument(page: Page, filename: string) {
  try {
    await page.goto('/upload');
    await uploadDocument(page, filename, { waitForAnalysis: false });
    console.log(`✅ Test document uploaded: ${filename}`);
  } catch (error) {
    console.warn(`⚠️ Failed to upload test document ${filename}:`, error);
  }
}