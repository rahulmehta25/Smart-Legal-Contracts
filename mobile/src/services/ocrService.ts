import TextRecognition from 'react-native-text-recognition';
import { ScanResult, BoundingBox } from '@types/index';
import ImageCropPicker from 'react-native-image-crop-picker';

class OCRService {
  async extractTextFromImage(imagePath: string): Promise<ScanResult> {
    try {
      const startTime = Date.now();
      
      // Process image with text recognition
      const result = await TextRecognition.recognize(imagePath);
      
      const processingTime = Date.now() - startTime;
      
      // Convert recognition result to our format
      const boundingBoxes: BoundingBox[] = result.blocks?.map((block: any, index: number) => ({
        x: block.frame?.x || 0,
        y: block.frame?.y || 0,
        width: block.frame?.width || 0,
        height: block.frame?.height || 0,
        text: block.text || '',
        confidence: block.confidence || 0,
      })) || [];

      // Combine all text
      const text = result.blocks?.map((block: any) => block.text).join('\n') || '';
      
      // Calculate overall confidence
      const confidence = boundingBoxes.length > 0 
        ? boundingBoxes.reduce((sum, box) => sum + box.confidence, 0) / boundingBoxes.length 
        : 0;

      return {
        text,
        confidence,
        boundingBoxes,
        imageUri: imagePath,
        processingTime,
      };
    } catch (error) {
      console.error('Error extracting text from image:', error);
      throw new Error('Failed to extract text from image');
    }
  }

  async extractTextFromFrame(frame: any): Promise<ScanResult | null> {
    try {
      // Convert frame to image path (this would be implementation-specific)
      // For now, we'll return null as this requires camera frame processing
      return null;
    } catch (error) {
      console.error('Error extracting text from frame:', error);
      return null;
    }
  }

  async preprocessImage(imagePath: string): Promise<string> {
    try {
      // Apply image preprocessing to improve OCR accuracy
      const processedImage = await ImageCropPicker.openCropper({
        path: imagePath,
        width: 2000,
        height: 2000,
        cropping: false,
        includeBase64: false,
        compressImageQuality: 0.9,
        freeStyleCropEnabled: true,
      });

      return processedImage.path;
    } catch (error) {
      console.error('Error preprocessing image:', error);
      // Return original path if preprocessing fails
      return imagePath;
    }
  }

  async enhanceImageForOCR(imagePath: string): Promise<string> {
    try {
      // Apply enhancements specifically for OCR
      // This could include:
      // - Contrast adjustment
      // - Noise reduction
      // - Perspective correction
      // - Sharpening
      
      // For now, return the original path
      // In a real implementation, you'd use image processing libraries
      return imagePath;
    } catch (error) {
      console.error('Error enhancing image for OCR:', error);
      return imagePath;
    }
  }

  async detectDocumentBounds(imagePath: string): Promise<{
    corners: Array<{ x: number; y: number }>;
    confidence: number;
  } | null> {
    try {
      // Document boundary detection
      // This would use computer vision to detect document edges
      // For now, return null (not implemented)
      return null;
    } catch (error) {
      console.error('Error detecting document bounds:', error);
      return null;
    }
  }

  async correctPerspective(
    imagePath: string, 
    corners: Array<{ x: number; y: number }>
  ): Promise<string> {
    try {
      // Apply perspective correction based on detected corners
      // This would transform the image to a rectangular view
      // For now, return the original path
      return imagePath;
    } catch (error) {
      console.error('Error correcting perspective:', error);
      return imagePath;
    }
  }

  async batchProcessImages(imagePaths: string[]): Promise<ScanResult[]> {
    try {
      const results: ScanResult[] = [];
      
      for (const imagePath of imagePaths) {
        try {
          const result = await this.extractTextFromImage(imagePath);
          results.push(result);
        } catch (error) {
          console.error(`Error processing image ${imagePath}:`, error);
          // Continue with other images
        }
      }
      
      return results;
    } catch (error) {
      console.error('Error in batch processing:', error);
      throw new Error('Failed to process images');
    }
  }

  async validateTextQuality(scanResult: ScanResult): Promise<{
    isValid: boolean;
    issues: string[];
    suggestions: string[];
  }> {
    const issues: string[] = [];
    const suggestions: string[] = [];

    // Check text length
    if (scanResult.text.length < 10) {
      issues.push('Very little text detected');
      suggestions.push('Ensure the document is fully visible and well-lit');
    }

    // Check confidence
    if (scanResult.confidence < 0.5) {
      issues.push('Low confidence in text recognition');
      suggestions.push('Try better lighting or hold the device steadier');
    }

    // Check for garbled text (too many special characters)
    const specialCharRatio = (scanResult.text.match(/[^a-zA-Z0-9\s]/g) || []).length / scanResult.text.length;
    if (specialCharRatio > 0.3) {
      issues.push('Text may be garbled or unclear');
      suggestions.push('Ensure the document is in focus and not blurred');
    }

    // Check bounding boxes
    if (scanResult.boundingBoxes.length === 0) {
      issues.push('No text regions detected');
      suggestions.push('Make sure the document fills most of the frame');
    }

    return {
      isValid: issues.length === 0,
      issues,
      suggestions,
    };
  }

  async getOptimalScanSettings(): Promise<{
    imageQuality: number;
    compressionLevel: number;
    recommendedResolution: { width: number; height: number };
  }> {
    return {
      imageQuality: 0.9,
      compressionLevel: 0.8,
      recommendedResolution: { width: 1920, height: 1080 },
    };
  }

  async extractStructuredData(text: string): Promise<{
    dates: string[];
    amounts: string[];
    emails: string[];
    phoneNumbers: string[];
    addresses: string[];
  }> {
    try {
      const dates = this.extractDates(text);
      const amounts = this.extractAmounts(text);
      const emails = this.extractEmails(text);
      const phoneNumbers = this.extractPhoneNumbers(text);
      const addresses = this.extractAddresses(text);

      return {
        dates,
        amounts,
        emails,
        phoneNumbers,
        addresses,
      };
    } catch (error) {
      console.error('Error extracting structured data:', error);
      return {
        dates: [],
        amounts: [],
        emails: [],
        phoneNumbers: [],
        addresses: [],
      };
    }
  }

  private extractDates(text: string): string[] {
    const datePatterns = [
      /\d{1,2}\/\d{1,2}\/\d{4}/g,
      /\d{1,2}-\d{1,2}-\d{4}/g,
      /\d{4}-\d{1,2}-\d{1,2}/g,
      /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}/gi,
    ];

    const dates: string[] = [];
    datePatterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        dates.push(...matches);
      }
    });

    return [...new Set(dates)]; // Remove duplicates
  }

  private extractAmounts(text: string): string[] {
    const amountPatterns = [
      /\$[\d,]+\.?\d*/g,
      /\b\d+\.?\d*\s*(?:USD|dollars?)\b/gi,
    ];

    const amounts: string[] = [];
    amountPatterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        amounts.push(...matches);
      }
    });

    return [...new Set(amounts)];
  }

  private extractEmails(text: string): string[] {
    const emailPattern = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
    return text.match(emailPattern) || [];
  }

  private extractPhoneNumbers(text: string): string[] {
    const phonePatterns = [
      /\(\d{3}\)\s*\d{3}-\d{4}/g,
      /\d{3}-\d{3}-\d{4}/g,
      /\d{3}\.\d{3}\.\d{4}/g,
      /\d{10}/g,
    ];

    const phones: string[] = [];
    phonePatterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        phones.push(...matches);
      }
    });

    return [...new Set(phones)];
  }

  private extractAddresses(text: string): string[] {
    // Simple address pattern - this could be much more sophisticated
    const addressPattern = /\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b/gi;
    return text.match(addressPattern) || [];
  }
}

export const ocrService = new OCRService();