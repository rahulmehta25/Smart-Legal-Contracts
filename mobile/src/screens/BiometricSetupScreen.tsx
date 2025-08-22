import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { biometricService } from '@services/biometricService';

const BiometricSetupScreen: React.FC = () => {
  const { theme } = useTheme();
  const navigation = useNavigation();
  
  const [biometricType, setBiometricType] = useState<string>('');
  const [isSetupInProgress, setIsSetupInProgress] = useState(false);
  const [isAvailable, setIsAvailable] = useState(false);

  useEffect(() => {
    checkBiometricAvailability();
  }, []);

  const checkBiometricAvailability = async () => {
    try {
      const available = await biometricService.isBiometricAvailable();
      const type = await biometricService.getBiometricType();
      
      setIsAvailable(available);
      setBiometricType(type);
      
      if (!available) {
        Alert.alert(
          'Biometric Authentication Unavailable',
          'Your device does not support biometric authentication or it is not set up.',
          [{ text: 'OK', onPress: () => navigation.goBack() }]
        );
      }
    } catch (error) {
      console.error('Error checking biometric availability:', error);
      Alert.alert('Error', 'Failed to check biometric availability');
    }
  };

  const setupBiometric = async () => {
    try {
      setIsSetupInProgress(true);
      
      const success = await biometricService.setupBiometric();
      
      if (success) {
        Alert.alert(
          'Setup Complete',
          `${biometricType} authentication has been successfully enabled for this app.`,
          [
            {
              text: 'Done',
              onPress: () => navigation.goBack(),
            },
          ]
        );
      } else {
        Alert.alert(
          'Setup Failed',
          'Biometric authentication setup was cancelled or failed. You can try again later from Settings.',
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('Error setting up biometric:', error);
      Alert.alert(
        'Setup Error',
        'An error occurred while setting up biometric authentication. Please try again.'
      );
    } finally {
      setIsSetupInProgress(false);
    }
  };

  const getBiometricIcon = () => {
    switch (biometricType.toLowerCase()) {
      case 'face id':
      case 'face':
        return 'face';
      case 'touch id':
      case 'fingerprint':
        return 'fingerprint';
      default:
        return 'security';
    }
  };

  const getBiometricDescription = () => {
    switch (biometricType.toLowerCase()) {
      case 'face id':
      case 'face':
        return 'Use Face ID to quickly and securely access your documents and analysis results.';
      case 'touch id':
      case 'fingerprint':
        return 'Use your fingerprint to quickly and securely access your documents and analysis results.';
      default:
        return 'Use biometric authentication to quickly and securely access your documents and analysis results.';
    }
  };

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
      padding: theme.spacing.xl,
      justifyContent: 'center',
    },
    iconContainer: {
      alignItems: 'center',
      marginBottom: theme.spacing.xl,
    },
    biometricIcon: {
      width: 120,
      height: 120,
      borderRadius: 60,
      backgroundColor: theme.colors.primary + '20',
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.lg,
    },
    title: {
      fontSize: theme.typography.fontSize.xxl,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      textAlign: 'center',
      marginBottom: theme.spacing.md,
    },
    subtitle: {
      fontSize: theme.typography.fontSize.lg,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      marginBottom: theme.spacing.xl,
    },
    description: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
      textAlign: 'center',
      lineHeight: 24,
      marginBottom: theme.spacing.xl,
    },
    benefitsContainer: {
      marginBottom: theme.spacing.xl,
    },
    benefitItem: {
      flexDirection: 'row',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    benefitIcon: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: theme.colors.success + '20',
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    benefitText: {
      flex: 1,
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.text,
    },
    buttonContainer: {
      marginTop: theme.spacing.xl,
    },
    setupButton: {
      backgroundColor: theme.colors.primary,
      paddingVertical: theme.spacing.lg,
      borderRadius: theme.borderRadius.lg,
      alignItems: 'center',
      marginBottom: theme.spacing.md,
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 8,
      elevation: 4,
    },
    setupButtonText: {
      color: 'white',
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
    },
    skipButton: {
      paddingVertical: theme.spacing.md,
      alignItems: 'center',
    },
    skipButtonText: {
      color: theme.colors.textSecondary,
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
    loadingContainer: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'center',
    },
    loadingText: {
      color: 'white',
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      marginLeft: theme.spacing.md,
    },
  });

  if (!isAvailable) {
    return (
      <View style={styles.container}>
        <View style={styles.iconContainer}>
          <View style={styles.biometricIcon}>
            <Icon name="error-outline" size={60} color={theme.colors.error} />
          </View>
          <Text style={styles.title}>Not Available</Text>
          <Text style={styles.description}>
            Biometric authentication is not available on this device or has not been set up in your device settings.
          </Text>
        </View>
        
        <TouchableOpacity
          style={[styles.setupButton, { backgroundColor: theme.colors.textSecondary }]}
          onPress={() => navigation.goBack()}
        >
          <Text style={styles.setupButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.iconContainer}>
        <View style={styles.biometricIcon}>
          <Icon name={getBiometricIcon()} size={60} color={theme.colors.primary} />
        </View>
        <Text style={styles.title}>Enable {biometricType}</Text>
        <Text style={styles.subtitle}>Secure and convenient access</Text>
      </View>

      <Text style={styles.description}>
        {getBiometricDescription()}
      </Text>

      <View style={styles.benefitsContainer}>
        <View style={styles.benefitItem}>
          <View style={styles.benefitIcon}>
            <Icon name="speed" size={20} color={theme.colors.success} />
          </View>
          <Text style={styles.benefitText}>
            Quick access to your documents without entering passwords
          </Text>
        </View>
        
        <View style={styles.benefitItem}>
          <View style={styles.benefitIcon}>
            <Icon name="security" size={20} color={theme.colors.success} />
          </View>
          <Text style={styles.benefitText}>
            Enhanced security using your unique biometric data
          </Text>
        </View>
        
        <View style={styles.benefitItem}>
          <View style={styles.benefitIcon}>
            <Icon name="privacy-tip" size={20} color={theme.colors.success} />
          </View>
          <Text style={styles.benefitText}>
            Your biometric data stays secure on your device
          </Text>
        </View>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.setupButton}
          onPress={setupBiometric}
          disabled={isSetupInProgress}
        >
          {isSetupInProgress ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator color="white" size="small" />
              <Text style={styles.loadingText}>Setting up...</Text>
            </View>
          ) : (
            <Text style={styles.setupButtonText}>
              Enable {biometricType}
            </Text>
          )}
        </TouchableOpacity>
        
        <TouchableOpacity
          style={styles.skipButton}
          onPress={() => navigation.goBack()}
          disabled={isSetupInProgress}
        >
          <Text style={styles.skipButtonText}>Skip for now</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

export default BiometricSetupScreen;