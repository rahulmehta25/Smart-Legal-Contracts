/**
 * Hook for offline mode functionality
 */

import { useState, useEffect, useCallback } from 'react';
import ArbitrationSDK from '../ArbitrationSDK';
import { OfflineSyncStatus, UseOfflineModeResult } from '../types';

export function useOfflineMode(): UseOfflineModeResult {
  const [isOffline, setIsOffline] = useState(false);
  const [syncStatus, setSyncStatus] = useState<OfflineSyncStatus>({
    isOnline: true,
    pendingUploads: 0,
    syncInProgress: false
  });

  useEffect(() => {
    const sdk = ArbitrationSDK.getInstance();

    const updateSyncStatus = async () => {
      try {
        const status = await sdk.getSyncStatus();
        setSyncStatus(status);
        setIsOffline(!status.isOnline);
      } catch (error) {
        console.warn('Failed to get sync status:', error);
      }
    };

    const handleNetworkStateChanged = (networkInfo: any) => {
      setIsOffline(!networkInfo.isConnected);
    };

    const handleSyncStatusChanged = (status: OfflineSyncStatus) => {
      setSyncStatus(status);
    };

    const handleOfflineModeChanged = (enabled: boolean) => {
      // Update local state if needed
    };

    // Setup event listeners
    sdk.on('networkStateChanged', handleNetworkStateChanged);
    sdk.on('syncStatusChanged', handleSyncStatusChanged);
    sdk.on('offlineModeChanged', handleOfflineModeChanged);

    // Initial status check
    updateSyncStatus();

    // Periodic status updates
    const interval = setInterval(updateSyncStatus, 30000); // Every 30 seconds

    return () => {
      sdk.off('networkStateChanged', handleNetworkStateChanged);
      sdk.off('syncStatusChanged', handleSyncStatusChanged);
      sdk.off('offlineModeChanged', handleOfflineModeChanged);
      clearInterval(interval);
    };
  }, []);

  const sync = useCallback(async (): Promise<void> => {
    const sdk = ArbitrationSDK.getInstance();
    
    try {
      await sdk.syncOfflineData();
    } catch (error) {
      throw new Error(`Sync failed: ${error}`);
    }
  }, []);

  const enableOfflineMode = useCallback((enabled: boolean): void => {
    const sdk = ArbitrationSDK.getInstance();
    sdk.enableOfflineMode(enabled);
  }, []);

  return {
    isOffline,
    syncStatus,
    sync,
    enableOfflineMode
  };
}