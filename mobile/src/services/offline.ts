import SQLite from 'react-native-sqlite-storage';
import NetInfo from '@react-native-community/netinfo';
import { Document, ArbitrationAnalysis, SyncQueueItem, SyncStatus } from '@types/index';

SQLite.DEBUG(false);
SQLite.enablePromise(true);

class OfflineService {
  private db: SQLite.SQLiteDatabase | null = null;
  private syncQueue: SyncQueueItem[] = [];
  private isOnline = false;
  private syncInProgress = false;

  constructor() {
    this.initializeDatabase();
    this.setupNetworkListener();
  }

  private async initializeDatabase(): Promise<void> {
    try {
      this.db = await SQLite.openDatabase({
        name: 'ArbitrationDetector.db',
        location: 'default',
      });

      await this.createTables();
      await this.loadSyncQueue();
    } catch (error) {
      console.error('Error initializing database:', error);
      throw error;
    }
  }

  private async createTables(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    const queries = [
      // Documents table
      `CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        content TEXT,
        extractedText TEXT,
        createdAt TEXT NOT NULL,
        updatedAt TEXT NOT NULL,
        imagePath TEXT,
        size INTEGER,
        type TEXT,
        analysisStatus TEXT,
        syncStatus TEXT DEFAULT 'pending'
      )`,

      // Analyses table
      `CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        documentId TEXT NOT NULL,
        hasArbitrationClause INTEGER,
        confidence REAL,
        riskLevel TEXT,
        recommendations TEXT,
        detectedClauses TEXT,
        createdAt TEXT NOT NULL,
        processingTime INTEGER,
        syncStatus TEXT DEFAULT 'pending',
        FOREIGN KEY (documentId) REFERENCES documents (id)
      )`,

      // Sync queue table
      `CREATE TABLE IF NOT EXISTS sync_queue (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        action TEXT NOT NULL,
        data TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        retryCount INTEGER DEFAULT 0,
        maxRetries INTEGER DEFAULT 3
      )`,

      // User preferences table
      `CREATE TABLE IF NOT EXISTS user_preferences (
        id TEXT PRIMARY KEY,
        preferences TEXT NOT NULL,
        updatedAt TEXT NOT NULL,
        syncStatus TEXT DEFAULT 'pending'
      )`,
    ];

    for (const query of queries) {
      await this.db.executeSql(query);
    }
  }

  private setupNetworkListener(): void {
    NetInfo.addEventListener(state => {
      const wasOnline = this.isOnline;
      this.isOnline = state.isConnected ?? false;

      if (!wasOnline && this.isOnline && this.syncQueue.length > 0) {
        this.processSyncQueue();
      }
    });
  }

  private async loadSyncQueue(): Promise<void> {
    if (!this.db) return;

    try {
      const [results] = await this.db.executeSql(
        'SELECT * FROM sync_queue ORDER BY timestamp ASC'
      );

      this.syncQueue = [];
      for (let i = 0; i < results.rows.length; i++) {
        const row = results.rows.item(i);
        this.syncQueue.push({
          id: row.id,
          type: row.type,
          action: row.action,
          data: JSON.parse(row.data),
          timestamp: new Date(row.timestamp),
          retryCount: row.retryCount,
          maxRetries: row.maxRetries,
        });
      }
    } catch (error) {
      console.error('Error loading sync queue:', error);
    }
  }

  // Document operations
  async saveDocument(document: Document): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const syncStatus = this.isOnline ? SyncStatus.PENDING : SyncStatus.OFFLINE;
      
      await this.db.executeSql(
        `INSERT OR REPLACE INTO documents 
         (id, name, content, extractedText, createdAt, updatedAt, imagePath, size, type, analysisStatus, syncStatus)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          document.id,
          document.name,
          document.content,
          document.extractedText,
          document.createdAt.toISOString(),
          document.updatedAt.toISOString(),
          document.imagePath || null,
          document.size,
          document.type,
          document.analysisStatus,
          syncStatus,
        ]
      );

      if (this.isOnline) {
        await this.addToSyncQueue('document', 'create', document);
      }
    } catch (error) {
      console.error('Error saving document:', error);
      throw error;
    }
  }

  async getDocument(id: string): Promise<Document | null> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const [results] = await this.db.executeSql(
        'SELECT * FROM documents WHERE id = ?',
        [id]
      );

      if (results.rows.length === 0) return null;

      const row = results.rows.item(0);
      return {
        id: row.id,
        name: row.name,
        content: row.content,
        extractedText: row.extractedText,
        createdAt: new Date(row.createdAt),
        updatedAt: new Date(row.updatedAt),
        imagePath: row.imagePath,
        size: row.size,
        type: row.type,
        analysisStatus: row.analysisStatus,
        syncStatus: row.syncStatus,
      };
    } catch (error) {
      console.error('Error getting document:', error);
      throw error;
    }
  }

  async getAllDocuments(): Promise<Document[]> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const [results] = await this.db.executeSql(
        'SELECT * FROM documents ORDER BY createdAt DESC'
      );

      const documents: Document[] = [];
      for (let i = 0; i < results.rows.length; i++) {
        const row = results.rows.item(i);
        documents.push({
          id: row.id,
          name: row.name,
          content: row.content,
          extractedText: row.extractedText,
          createdAt: new Date(row.createdAt),
          updatedAt: new Date(row.updatedAt),
          imagePath: row.imagePath,
          size: row.size,
          type: row.type,
          analysisStatus: row.analysisStatus,
          syncStatus: row.syncStatus,
        });
      }

      return documents;
    } catch (error) {
      console.error('Error getting all documents:', error);
      throw error;
    }
  }

  async deleteDocument(id: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      await this.db.executeSql('DELETE FROM documents WHERE id = ?', [id]);
      await this.db.executeSql('DELETE FROM analyses WHERE documentId = ?', [id]);

      if (this.isOnline) {
        await this.addToSyncQueue('document', 'delete', { id });
      }
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error;
    }
  }

  // Analysis operations
  async saveAnalysis(analysis: ArbitrationAnalysis): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const syncStatus = this.isOnline ? SyncStatus.PENDING : SyncStatus.OFFLINE;

      await this.db.executeSql(
        `INSERT OR REPLACE INTO analyses 
         (id, documentId, hasArbitrationClause, confidence, riskLevel, recommendations, detectedClauses, createdAt, processingTime, syncStatus)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          analysis.id,
          analysis.documentId,
          analysis.hasArbitrationClause ? 1 : 0,
          analysis.confidence,
          analysis.riskLevel,
          JSON.stringify(analysis.recommendations),
          JSON.stringify(analysis.detectedClauses),
          analysis.createdAt.toISOString(),
          analysis.processingTime,
          syncStatus,
        ]
      );

      if (this.isOnline) {
        await this.addToSyncQueue('analysis', 'create', analysis);
      }
    } catch (error) {
      console.error('Error saving analysis:', error);
      throw error;
    }
  }

  async getAnalysis(documentId: string): Promise<ArbitrationAnalysis | null> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const [results] = await this.db.executeSql(
        'SELECT * FROM analyses WHERE documentId = ?',
        [documentId]
      );

      if (results.rows.length === 0) return null;

      const row = results.rows.item(0);
      return {
        id: row.id,
        documentId: row.documentId,
        hasArbitrationClause: row.hasArbitrationClause === 1,
        confidence: row.confidence,
        riskLevel: row.riskLevel,
        recommendations: JSON.parse(row.recommendations),
        detectedClauses: JSON.parse(row.detectedClauses),
        createdAt: new Date(row.createdAt),
        processingTime: row.processingTime,
      };
    } catch (error) {
      console.error('Error getting analysis:', error);
      throw error;
    }
  }

  async getAllAnalyses(): Promise<ArbitrationAnalysis[]> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      const [results] = await this.db.executeSql(
        'SELECT * FROM analyses ORDER BY createdAt DESC'
      );

      const analyses: ArbitrationAnalysis[] = [];
      for (let i = 0; i < results.rows.length; i++) {
        const row = results.rows.item(i);
        analyses.push({
          id: row.id,
          documentId: row.documentId,
          hasArbitrationClause: row.hasArbitrationClause === 1,
          confidence: row.confidence,
          riskLevel: row.riskLevel,
          recommendations: JSON.parse(row.recommendations),
          detectedClauses: JSON.parse(row.detectedClauses),
          createdAt: new Date(row.createdAt),
          processingTime: row.processingTime,
        });
      }

      return analyses;
    } catch (error) {
      console.error('Error getting all analyses:', error);
      throw error;
    }
  }

  // Sync queue operations
  private async addToSyncQueue(
    type: SyncQueueItem['type'],
    action: SyncQueueItem['action'],
    data: any
  ): Promise<void> {
    if (!this.db) return;

    const item: SyncQueueItem = {
      id: `${type}_${action}_${Date.now()}_${Math.random()}`,
      type,
      action,
      data,
      timestamp: new Date(),
      retryCount: 0,
      maxRetries: 3,
    };

    try {
      await this.db.executeSql(
        'INSERT INTO sync_queue (id, type, action, data, timestamp, retryCount, maxRetries) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [
          item.id,
          item.type,
          item.action,
          JSON.stringify(item.data),
          item.timestamp.toISOString(),
          item.retryCount,
          item.maxRetries,
        ]
      );

      this.syncQueue.push(item);

      if (this.isOnline && !this.syncInProgress) {
        this.processSyncQueue();
      }
    } catch (error) {
      console.error('Error adding to sync queue:', error);
    }
  }

  private async processSyncQueue(): Promise<void> {
    if (!this.isOnline || this.syncInProgress || this.syncQueue.length === 0) {
      return;
    }

    this.syncInProgress = true;

    try {
      const itemsToProcess = [...this.syncQueue];

      for (const item of itemsToProcess) {
        try {
          const success = await this.syncItem(item);

          if (success) {
            await this.removeSyncQueueItem(item.id);
          } else {
            await this.incrementRetryCount(item.id);
          }
        } catch (error) {
          console.error('Error syncing item:', error);
          await this.incrementRetryCount(item.id);
        }
      }
    } finally {
      this.syncInProgress = false;
    }
  }

  private async syncItem(item: SyncQueueItem): Promise<boolean> {
    // This would integrate with your backend API
    // For now, we'll simulate the sync process
    
    try {
      switch (item.type) {
        case 'document':
          return await this.syncDocument(item);
        case 'analysis':
          return await this.syncAnalysis(item);
        case 'user_preferences':
          return await this.syncUserPreferences(item);
        default:
          return false;
      }
    } catch (error) {
      console.error('Error in syncItem:', error);
      return false;
    }
  }

  private async syncDocument(item: SyncQueueItem): Promise<boolean> {
    // Implement actual API calls here
    // This is a placeholder implementation
    
    switch (item.action) {
      case 'create':
        // POST to /api/documents
        console.log('Syncing document creation:', item.data);
        return true;
      case 'update':
        // PUT to /api/documents/:id
        console.log('Syncing document update:', item.data);
        return true;
      case 'delete':
        // DELETE to /api/documents/:id
        console.log('Syncing document deletion:', item.data);
        return true;
      default:
        return false;
    }
  }

  private async syncAnalysis(item: SyncQueueItem): Promise<boolean> {
    // Implement actual API calls here
    switch (item.action) {
      case 'create':
        // POST to /api/analyses
        console.log('Syncing analysis creation:', item.data);
        return true;
      case 'update':
        // PUT to /api/analyses/:id
        console.log('Syncing analysis update:', item.data);
        return true;
      default:
        return false;
    }
  }

  private async syncUserPreferences(item: SyncQueueItem): Promise<boolean> {
    // Implement actual API calls here
    console.log('Syncing user preferences:', item.data);
    return true;
  }

  private async removeSyncQueueItem(id: string): Promise<void> {
    if (!this.db) return;

    try {
      await this.db.executeSql('DELETE FROM sync_queue WHERE id = ?', [id]);
      this.syncQueue = this.syncQueue.filter(item => item.id !== id);
    } catch (error) {
      console.error('Error removing sync queue item:', error);
    }
  }

  private async incrementRetryCount(id: string): Promise<void> {
    if (!this.db) return;

    try {
      const [results] = await this.db.executeSql(
        'SELECT retryCount, maxRetries FROM sync_queue WHERE id = ?',
        [id]
      );

      if (results.rows.length > 0) {
        const row = results.rows.item(0);
        const newRetryCount = row.retryCount + 1;

        if (newRetryCount >= row.maxRetries) {
          // Max retries reached, remove from queue
          await this.removeSyncQueueItem(id);
        } else {
          // Increment retry count
          await this.db.executeSql(
            'UPDATE sync_queue SET retryCount = ? WHERE id = ?',
            [newRetryCount, id]
          );

          // Update local queue
          const item = this.syncQueue.find(i => i.id === id);
          if (item) {
            item.retryCount = newRetryCount;
          }
        }
      }
    } catch (error) {
      console.error('Error incrementing retry count:', error);
    }
  }

  // Utility methods
  async getSyncStatus(): Promise<{
    pendingItems: number;
    isOnline: boolean;
    lastSyncTime?: Date;
  }> {
    return {
      pendingItems: this.syncQueue.length,
      isOnline: this.isOnline,
      lastSyncTime: this.syncQueue.length > 0 ? undefined : new Date(),
    };
  }

  async forceSyncNow(): Promise<void> {
    if (this.isOnline) {
      await this.processSyncQueue();
    } else {
      throw new Error('Cannot sync while offline');
    }
  }

  async clearLocalData(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');

    try {
      await this.db.executeSql('DELETE FROM documents');
      await this.db.executeSql('DELETE FROM analyses');
      await this.db.executeSql('DELETE FROM sync_queue');
      await this.db.executeSql('DELETE FROM user_preferences');
      
      this.syncQueue = [];
    } catch (error) {
      console.error('Error clearing local data:', error);
      throw error;
    }
  }

  async closeDatabase(): Promise<void> {
    if (this.db) {
      await this.db.close();
      this.db = null;
    }
  }
}

export const offlineService = new OfflineService();