import ReactNativeBiometrics from 'react-native-biometrics';
import * as Keychain from 'react-native-keychain';
import { Alert } from 'react-native';

class BiometricService {
  private rnBiometrics: ReactNativeBiometrics;

  constructor() {
    this.rnBiometrics = new ReactNativeBiometrics({
      allowDeviceCredentials: true,
    });
  }

  async isBiometricAvailable(): Promise<boolean> {
    try {
      const { available } = await this.rnBiometrics.isSensorAvailable();
      return available;
    } catch (error) {
      console.error('Error checking biometric availability:', error);
      return false;
    }
  }

  async getBiometricType(): Promise<string> {
    try {
      const { biometryType } = await this.rnBiometrics.isSensorAvailable();
      
      switch (biometryType) {
        case ReactNativeBiometrics.TouchID:
          return 'Touch ID';
        case ReactNativeBiometrics.FaceID:
          return 'Face ID';
        case ReactNativeBiometrics.Biometrics:
          return 'Fingerprint';
        default:
          return 'Biometric';
      }
    } catch (error) {
      console.error('Error getting biometric type:', error);
      return 'Biometric';
    }
  }

  async setupBiometric(): Promise<boolean> {
    try {
      const isAvailable = await this.isBiometricAvailable();
      if (!isAvailable) {
        throw new Error('Biometric authentication is not available on this device');
      }

      // Check if keys already exist
      const { keysExist } = await this.rnBiometrics.biometricKeysExist();
      
      if (!keysExist) {
        // Create biometric key pair
        const { publicKey } = await this.rnBiometrics.createKeys();
        
        // Store the public key securely
        await this.storeBiometricKey(publicKey);
      }

      // Test biometric authentication
      const authResult = await this.authenticateWithBiometric('Set up biometric authentication');
      
      if (authResult.success) {
        await this.enableBiometricAuth();
        return true;
      } else {
        return false;
      }
    } catch (error) {
      console.error('Error setting up biometric:', error);
      Alert.alert('Setup Error', 'Failed to set up biometric authentication');
      return false;
    }
  }

  async authenticateWithBiometric(promptMessage: string = 'Authenticate with biometric'): Promise<{
    success: boolean;
    signature?: string;
    error?: string;
  }> {
    try {
      const isAvailable = await this.isBiometricAvailable();
      if (!isAvailable) {
        return {
          success: false,
          error: 'Biometric authentication is not available',
        };
      }

      const { keysExist } = await this.rnBiometrics.biometricKeysExist();
      if (!keysExist) {
        return {
          success: false,
          error: 'Biometric authentication is not set up',
        };
      }

      // Create a signature payload
      const payload = `${Date.now()}-auth-payload`;
      
      const { success, signature, error } = await this.rnBiometrics.createSignature({
        promptMessage,
        payload,
        cancelButtonText: 'Cancel',
        fallbackPromptMessage: 'Use device passcode',
      });

      if (success && signature) {
        return {
          success: true,
          signature,
        };
      } else {
        return {
          success: false,
          error: error || 'Authentication failed',
        };
      }
    } catch (error) {
      console.error('Error during biometric authentication:', error);
      return {
        success: false,
        error: 'Authentication error occurred',
      };
    }
  }

  async disableBiometric(): Promise<boolean> {
    try {
      // Delete biometric keys
      const { keysDeleted } = await this.rnBiometrics.deleteKeys();
      
      if (keysDeleted) {
        await this.disableBiometricAuth();
        await this.removeBiometricKey();
        return true;
      } else {
        return false;
      }
    } catch (error) {
      console.error('Error disabling biometric:', error);
      return false;
    }
  }

  async isBiometricEnabled(): Promise<boolean> {
    try {
      const result = await Keychain.getInternetCredentials('biometric_enabled');
      return result && result.password === 'true';
    } catch (error) {
      console.error('Error checking if biometric is enabled:', error);
      return false;
    }
  }

  private async enableBiometricAuth(): Promise<void> {
    try {
      await Keychain.setInternetCredentials(
        'biometric_enabled',
        'biometric',
        'true',
        {
          accessControl: Keychain.ACCESS_CONTROL.BIOMETRY_ANY,
          authenticationType: Keychain.AUTHENTICATION_TYPE.BIOMETRICS,
        }
      );
    } catch (error) {
      console.error('Error enabling biometric auth:', error);
      throw error;
    }
  }

  private async disableBiometricAuth(): Promise<void> {
    try {
      await Keychain.resetInternetCredentials('biometric_enabled');
    } catch (error) {
      console.error('Error disabling biometric auth:', error);
      throw error;
    }
  }

  private async storeBiometricKey(publicKey: string): Promise<void> {
    try {
      await Keychain.setInternetCredentials(
        'biometric_public_key',
        'key',
        publicKey
      );
    } catch (error) {
      console.error('Error storing biometric key:', error);
      throw error;
    }
  }

  private async removeBiometricKey(): Promise<void> {
    try {
      await Keychain.resetInternetCredentials('biometric_public_key');
    } catch (error) {
      console.error('Error removing biometric key:', error);
      // Don't throw - this is cleanup
    }
  }

  async getBiometricPublicKey(): Promise<string | null> {
    try {
      const result = await Keychain.getInternetCredentials('biometric_public_key');
      return result ? result.password : null;
    } catch (error) {
      console.error('Error getting biometric public key:', error);
      return null;
    }
  }

  async authenticateForAppAccess(): Promise<boolean> {
    try {
      const isEnabled = await this.isBiometricEnabled();
      if (!isEnabled) {
        return true; // If biometric is not enabled, allow access
      }

      const biometricType = await this.getBiometricType();
      const result = await this.authenticateWithBiometric(
        `Use ${biometricType} to access Arbitration Detector`
      );

      return result.success;
    } catch (error) {
      console.error('Error authenticating for app access:', error);
      return false;
    }
  }

  async authenticateForSensitiveAction(action: string): Promise<boolean> {
    try {
      const isEnabled = await this.isBiometricEnabled();
      if (!isEnabled) {
        // If biometric is not enabled, you might want to prompt for device passcode
        // or allow the action without additional auth
        return true;
      }

      const biometricType = await this.getBiometricType();
      const result = await this.authenticateWithBiometric(
        `Use ${biometricType} to ${action}`
      );

      return result.success;
    } catch (error) {
      console.error('Error authenticating for sensitive action:', error);
      return false;
    }
  }

  async verifySignature(signature: string, payload: string): Promise<boolean> {
    try {
      // In a real implementation, you would verify the signature
      // against the stored public key using cryptographic functions
      // For now, we'll just return true if signature exists
      return signature && signature.length > 0;
    } catch (error) {
      console.error('Error verifying signature:', error);
      return false;
    }
  }

  async getBiometricCapabilities(): Promise<{
    available: boolean;
    biometryType: string;
    keysExist: boolean;
    enabled: boolean;
  }> {
    try {
      const available = await this.isBiometricAvailable();
      const biometryType = await this.getBiometricType();
      const { keysExist } = await this.rnBiometrics.biometricKeysExist();
      const enabled = await this.isBiometricEnabled();

      return {
        available,
        biometryType,
        keysExist,
        enabled,
      };
    } catch (error) {
      console.error('Error getting biometric capabilities:', error);
      return {
        available: false,
        biometryType: 'Unknown',
        keysExist: false,
        enabled: false,
      };
    }
  }

  async resetBiometricSetup(): Promise<boolean> {
    try {
      // Delete existing keys
      await this.rnBiometrics.deleteKeys();
      
      // Clear stored credentials
      await this.disableBiometricAuth();
      await this.removeBiometricKey();
      
      return true;
    } catch (error) {
      console.error('Error resetting biometric setup:', error);
      return false;
    }
  }
}

export const biometricService = new BiometricService();