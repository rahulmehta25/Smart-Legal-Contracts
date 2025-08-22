/**
 * Hook for biometric authentication functionality
 */

import { useState, useEffect, useCallback } from 'react';
import ArbitrationSDK from '../ArbitrationSDK';
import { BiometricType, BiometricAuthResult, UseBiometricAuthResult } from '../types';

export function useBiometricAuth(): UseBiometricAuthResult {
  const [isAvailable, setIsAvailable] = useState(false);
  const [biometricType, setBiometricType] = useState<BiometricType>(BiometricType.NONE);
  const [isEnrolled, setIsEnrolled] = useState(false);

  useEffect(() => {
    const checkBiometricStatus = async () => {
      const sdk = ArbitrationSDK.getInstance();
      
      try {
        const available = await sdk.isBiometricAvailable();
        setIsAvailable(available);

        if (available) {
          const type = await sdk.getBiometricType();
          setBiometricType(type);
          
          // Check if user has enrolled biometrics
          // This would typically be checked via a native module method
          setIsEnrolled(type !== BiometricType.NONE);
        }
      } catch (error) {
        console.warn('Failed to check biometric status:', error);
        setIsAvailable(false);
        setBiometricType(BiometricType.NONE);
        setIsEnrolled(false);
      }
    };

    checkBiometricStatus();
  }, []);

  const authenticate = useCallback(async (reason: string): Promise<BiometricAuthResult> => {
    if (!isAvailable) {
      throw new Error('Biometric authentication is not available');
    }

    const sdk = ArbitrationSDK.getInstance();
    
    try {
      const result = await sdk.authenticateWithBiometrics(reason);
      return result;
    } catch (error) {
      throw new Error(`Authentication failed: ${error}`);
    }
  }, [isAvailable]);

  return {
    isAvailable,
    biometricType,
    authenticate,
    isEnrolled
  };
}