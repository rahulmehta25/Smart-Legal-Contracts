import { Document, AnalysisStatus, SyncStatus, ScanResult } from '@types/index';
import { offlineService } from './offline';
import RNFS from 'react-native-fs';
import { generateUUID } from '@utils/helpers';

class DocumentService {
  async createFromScan(scanResult: ScanResult): Promise<Document> {
    try {
      const documentId = generateUUID();
      const timestamp = new Date();
      
      // Save the image to local storage
      const imagePath = await this.saveImageToLocal(scanResult.imageUri, documentId);
      
      const document: Document = {
        id: documentId,
        name: this.generateDocumentName(scanResult.text),
        content: '', // Raw document content if available
        extractedText: scanResult.text,
        createdAt: timestamp,
        updatedAt: timestamp,
        imagePath,
        size: await this.getFileSize(imagePath),
        type: 'image',
        analysisStatus: AnalysisStatus.PENDING,
        syncStatus: SyncStatus.PENDING,
      };

      await offlineService.saveDocument(document);
      return document;
    } catch (error) {
      console.error('Error creating document from scan:', error);
      throw new Error('Failed to create document from scan');
    }
  }

  async createFromFile(filePath: string, fileName: string): Promise<Document> {
    try {
      const documentId = generateUUID();
      const timestamp = new Date();
      
      // Copy file to app's document directory
      const localPath = await this.copyFileToLocal(filePath, documentId);
      const fileSize = await this.getFileSize(localPath);
      const extractedText = await this.extractTextFromFile(localPath);
      
      const document: Document = {
        id: documentId,
        name: fileName,
        content: '', // Could read file content for text files
        extractedText,
        createdAt: timestamp,
        updatedAt: timestamp,
        imagePath: localPath,
        size: fileSize,
        type: this.getFileType(fileName),
        analysisStatus: AnalysisStatus.PENDING,
        syncStatus: SyncStatus.PENDING,
      };

      await offlineService.saveDocument(document);
      return document;
    } catch (error) {
      console.error('Error creating document from file:', error);
      throw new Error('Failed to create document from file');
    }
  }

  async getDocumentById(id: string): Promise<Document | null> {
    try {
      return await offlineService.getDocument(id);
    } catch (error) {
      console.error('Error getting document by ID:', error);
      throw new Error('Failed to retrieve document');
    }
  }

  async getAllDocuments(): Promise<Document[]> {
    try {
      return await offlineService.getAllDocuments();
    } catch (error) {
      console.error('Error getting all documents:', error);
      throw new Error('Failed to retrieve documents');
    }
  }

  async updateDocument(document: Document): Promise<Document> {
    try {
      const updatedDocument = {
        ...document,
        updatedAt: new Date(),
      };

      await offlineService.saveDocument(updatedDocument);
      return updatedDocument;
    } catch (error) {
      console.error('Error updating document:', error);
      throw new Error('Failed to update document');
    }
  }

  async deleteDocument(id: string): Promise<void> {
    try {
      // Get document to delete associated files
      const document = await this.getDocumentById(id);
      
      if (document?.imagePath) {
        await this.deleteLocalFile(document.imagePath);
      }

      await offlineService.deleteDocument(id);
    } catch (error) {
      console.error('Error deleting document:', error);
      throw new Error('Failed to delete document');
    }
  }

  async updateAnalysisStatus(documentId: string, status: AnalysisStatus): Promise<void> {
    try {
      const document = await this.getDocumentById(documentId);
      if (!document) {
        throw new Error('Document not found');
      }

      const updatedDocument = {
        ...document,
        analysisStatus: status,
        updatedAt: new Date(),
      };

      await offlineService.saveDocument(updatedDocument);
    } catch (error) {
      console.error('Error updating analysis status:', error);
      throw new Error('Failed to update analysis status');
    }
  }

  async searchDocuments(query: string): Promise<Document[]> {
    try {
      const allDocuments = await this.getAllDocuments();
      
      const searchTerms = query.toLowerCase().split(' ');
      
      return allDocuments.filter(document => {
        const searchableText = `${document.name} ${document.extractedText}`.toLowerCase();
        
        return searchTerms.every(term =>
          searchableText.includes(term)
        );
      });
    } catch (error) {
      console.error('Error searching documents:', error);
      throw new Error('Failed to search documents');
    }
  }

  async getDocumentsByStatus(status: AnalysisStatus): Promise<Document[]> {
    try {
      const allDocuments = await this.getAllDocuments();
      return allDocuments.filter(doc => doc.analysisStatus === status);
    } catch (error) {
      console.error('Error getting documents by status:', error);
      throw new Error('Failed to get documents by status');
    }
  }

  async getDocumentStatistics(): Promise<{
    total: number;
    analyzed: number;
    pending: number;
    failed: number;
  }> {
    try {
      const allDocuments = await this.getAllDocuments();
      
      return {
        total: allDocuments.length,
        analyzed: allDocuments.filter(d => d.analysisStatus === AnalysisStatus.COMPLETED).length,
        pending: allDocuments.filter(d => 
          d.analysisStatus === AnalysisStatus.PENDING || 
          d.analysisStatus === AnalysisStatus.PROCESSING
        ).length,
        failed: allDocuments.filter(d => d.analysisStatus === AnalysisStatus.FAILED).length,
      };
    } catch (error) {
      console.error('Error getting document statistics:', error);
      throw new Error('Failed to get document statistics');
    }
  }

  // Private helper methods
  private async saveImageToLocal(sourceUri: string, documentId: string): Promise<string> {
    try {
      const documentsDir = RNFS.DocumentDirectoryPath + '/documents';
      await RNFS.mkdir(documentsDir);
      
      const filename = `${documentId}.jpg`;
      const destPath = `${documentsDir}/${filename}`;
      
      await RNFS.copyFile(sourceUri, destPath);
      return destPath;
    } catch (error) {
      console.error('Error saving image to local storage:', error);
      throw error;
    }
  }

  private async copyFileToLocal(sourcePath: string, documentId: string): Promise<string> {
    try {
      const documentsDir = RNFS.DocumentDirectoryPath + '/documents';
      await RNFS.mkdir(documentsDir);
      
      const fileExtension = sourcePath.split('.').pop() || 'file';
      const filename = `${documentId}.${fileExtension}`;
      const destPath = `${documentsDir}/${filename}`;
      
      await RNFS.copyFile(sourcePath, destPath);
      return destPath;
    } catch (error) {
      console.error('Error copying file to local storage:', error);
      throw error;
    }
  }

  private async getFileSize(filePath: string): Promise<number> {
    try {
      const stat = await RNFS.stat(filePath);
      return stat.size;
    } catch (error) {
      console.error('Error getting file size:', error);
      return 0;
    }
  }

  private async deleteLocalFile(filePath: string): Promise<void> {
    try {
      const exists = await RNFS.exists(filePath);
      if (exists) {
        await RNFS.unlink(filePath);
      }
    } catch (error) {
      console.error('Error deleting local file:', error);
      // Don't throw - file deletion is not critical
    }
  }

  private generateDocumentName(extractedText: string): string {
    // Extract first meaningful line or create a name based on content
    const lines = extractedText.split('\n').filter(line => line.trim().length > 0);
    
    if (lines.length > 0) {
      let name = lines[0].trim();
      
      // Limit length and clean up
      if (name.length > 50) {
        name = name.substring(0, 47) + '...';
      }
      
      // Remove special characters that might cause issues
      name = name.replace(/[<>:"/\\|?*]/g, ' ').trim();
      
      if (name.length > 0) {
        return name;
      }
    }
    
    // Fallback to timestamp-based name
    const timestamp = new Date().toISOString().split('T')[0];
    return `Document ${timestamp}`;
  }

  private getFileType(fileName: string): string {
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    switch (extension) {
      case 'pdf':
        return 'pdf';
      case 'doc':
      case 'docx':
        return 'document';
      case 'txt':
        return 'text';
      case 'jpg':
      case 'jpeg':
      case 'png':
        return 'image';
      default:
        return 'unknown';
    }
  }

  private async extractTextFromFile(filePath: string): Promise<string> {
    try {
      const fileType = this.getFileType(filePath);
      
      switch (fileType) {
        case 'text':
          return await RNFS.readFile(filePath, 'utf8');
        case 'image':
          // Would use OCR service for images
          return ''; // Placeholder - actual OCR would happen here
        case 'pdf':
          // Would use PDF parsing library
          return ''; // Placeholder
        default:
          return '';
      }
    } catch (error) {
      console.error('Error extracting text from file:', error);
      return '';
    }
  }
}

export const documentService = new DocumentService();