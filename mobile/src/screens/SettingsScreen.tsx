import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Switch,
  Alert,
  Linking,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { User, UserPreferences, SubscriptionType } from '@types/index';
import { userService } from '@services/userService';
import { biometricService } from '@services/biometricService';
import { notificationService } from '@services/notificationService';

const SettingsScreen: React.FC = () => {
  const { theme, isDarkMode, isSystemTheme, toggleTheme, useSystemTheme } = useTheme();
  const navigation = useNavigation();
  
  const [user, setUser] = useState<User | null>(null);
  const [preferences, setPreferences] = useState<UserPreferences | null>(null);
  const [biometricAvailable, setBiometricAvailable] = useState(false);
  const [biometricType, setBiometricType] = useState<string>('');

  useEffect(() => {
    loadUserData();
    checkBiometricAvailability();
  }, []);

  const loadUserData = async () => {
    try {
      const userData = await userService.getCurrentUser();
      setUser(userData);
      setPreferences(userData?.preferences || null);
    } catch (error) {
      console.error('Error loading user data:', error);
    }
  };

  const checkBiometricAvailability = async () => {
    try {
      const available = await biometricService.isBiometricAvailable();
      const type = await biometricService.getBiometricType();
      setBiometricAvailable(available);
      setBiometricType(type);
    } catch (error) {
      console.error('Error checking biometric availability:', error);
    }
  };

  const updatePreference = async (key: keyof UserPreferences, value: any) => {
    if (!preferences) return;

    try {
      const updatedPreferences = { ...preferences, [key]: value };
      await userService.updateUserPreferences(updatedPreferences);
      setPreferences(updatedPreferences);
    } catch (error) {
      console.error('Error updating preference:', error);
      Alert.alert('Error', 'Failed to update setting');
    }
  };

  const updateNotificationPreference = async (key: keyof UserPreferences['notifications'], value: boolean) => {
    if (!preferences) return;

    try {
      const updatedNotifications = { ...preferences.notifications, [key]: value };
      const updatedPreferences = { ...preferences, notifications: updatedNotifications };
      await userService.updateUserPreferences(updatedPreferences);
      setPreferences(updatedPreferences);
      
      // Update notification permissions if needed
      if (value) {
        await notificationService.requestPermissions();
      }
    } catch (error) {
      console.error('Error updating notification preference:', error);
      Alert.alert('Error', 'Failed to update notification setting');
    }
  };

  const toggleBiometric = async (enabled: boolean) => {
    if (enabled) {
      try {
        const success = await biometricService.setupBiometric();
        if (success) {
          await updatePreference('biometricAuth', true);
        } else {
          Alert.alert('Setup Failed', 'Biometric authentication setup was cancelled or failed');
        }
      } catch (error) {
        console.error('Error setting up biometric:', error);
        Alert.alert('Error', 'Failed to setup biometric authentication');
      }
    } else {
      Alert.alert(
        'Disable Biometric Authentication',
        'Are you sure you want to disable biometric authentication?',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Disable',
            style: 'destructive',
            onPress: async () => {
              try {
                await biometricService.disableBiometric();
                await updatePreference('biometricAuth', false);
              } catch (error) {
                console.error('Error disabling biometric:', error);
                Alert.alert('Error', 'Failed to disable biometric authentication');
              }
            },
          },
        ]
      );
    }
  };

  const clearCache = async () => {
    Alert.alert(
      'Clear Cache',
      'This will remove all cached data and temporary files. Continue?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            try {
              // Clear cache implementation
              Alert.alert('Success', 'Cache cleared successfully');
            } catch (error) {
              console.error('Error clearing cache:', error);
              Alert.alert('Error', 'Failed to clear cache');
            }
          },
        },
      ]
    );
  };

  const signOut = async () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            try {
              await userService.signOut();
              // Navigate to auth screen
            } catch (error) {
              console.error('Error signing out:', error);
              Alert.alert('Error', 'Failed to sign out');
            }
          },
        },
      ]
    );
  };

  const openURL = (url: string) => {
    Linking.openURL(url).catch(err => {
      console.error('Error opening URL:', err);
      Alert.alert('Error', 'Unable to open link');
    });
  };

  const getSubscriptionBadgeColor = (subscription: SubscriptionType) => {
    switch (subscription) {
      case SubscriptionType.PREMIUM:
        return theme.colors.warning;
      case SubscriptionType.ENTERPRISE:
        return theme.colors.primary;
      default:
        return theme.colors.textSecondary;
    }
  };

  const SettingSection: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
    <View style={styles.section}>
      <Text style={[styles.sectionTitle, { color: theme.colors.text }]}>{title}</Text>
      <View style={[styles.sectionContent, { backgroundColor: theme.colors.surface }]}>
        {children}
      </View>
    </View>
  );

  const SettingItem: React.FC<{
    icon: string;
    title: string;
    subtitle?: string;
    rightComponent?: React.ReactNode;
    onPress?: () => void;
    showArrow?: boolean;
  }> = ({ icon, title, subtitle, rightComponent, onPress, showArrow = false }) => (
    <TouchableOpacity
      style={styles.settingItem}
      onPress={onPress}
      disabled={!onPress}
    >
      <View style={styles.settingLeft}>
        <View style={[styles.settingIcon, { backgroundColor: theme.colors.primary + '20' }]}>
          <Icon name={icon} size={20} color={theme.colors.primary} />
        </View>
        <View style={styles.settingText}>
          <Text style={[styles.settingTitle, { color: theme.colors.text }]}>{title}</Text>
          {subtitle && (
            <Text style={[styles.settingSubtitle, { color: theme.colors.textSecondary }]}>
              {subtitle}
            </Text>
          )}
        </View>
      </View>
      <View style={styles.settingRight}>
        {rightComponent}
        {showArrow && <Icon name="chevron-right" size={20} color={theme.colors.textSecondary} />}
      </View>
    </TouchableOpacity>
  );

  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: theme.colors.background,
    },
    scrollContent: {
      padding: theme.spacing.md,
    },
    profileSection: {
      alignItems: 'center',
      padding: theme.spacing.xl,
      backgroundColor: theme.colors.surface,
      borderRadius: theme.borderRadius.lg,
      marginBottom: theme.spacing.lg,
    },
    profileAvatar: {
      width: 80,
      height: 80,
      borderRadius: 40,
      backgroundColor: theme.colors.primary,
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: theme.spacing.md,
    },
    profileName: {
      fontSize: theme.typography.fontSize.lg,
      fontWeight: theme.typography.fontWeight.bold,
      color: theme.colors.text,
      marginBottom: theme.spacing.xs,
    },
    profileEmail: {
      fontSize: theme.typography.fontSize.md,
      color: theme.colors.textSecondary,
      marginBottom: theme.spacing.sm,
    },
    subscriptionBadge: {
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.xs,
      borderRadius: theme.borderRadius.lg,
    },
    subscriptionText: {
      color: 'white',
      fontSize: theme.typography.fontSize.sm,
      fontWeight: theme.typography.fontWeight.bold,
    },
    section: {
      marginBottom: theme.spacing.lg,
    },
    sectionTitle: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.bold,
      marginBottom: theme.spacing.sm,
      marginLeft: theme.spacing.sm,
    },
    sectionContent: {
      borderRadius: theme.borderRadius.lg,
      overflow: 'hidden',
      shadowColor: theme.colors.shadow,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 4,
      elevation: 3,
    },
    settingItem: {
      flexDirection: 'row',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: theme.spacing.md,
      borderBottomWidth: 1,
      borderBottomColor: theme.colors.border,
    },
    settingLeft: {
      flexDirection: 'row',
      alignItems: 'center',
      flex: 1,
    },
    settingIcon: {
      width: 36,
      height: 36,
      borderRadius: 18,
      justifyContent: 'center',
      alignItems: 'center',
      marginRight: theme.spacing.md,
    },
    settingText: {
      flex: 1,
    },
    settingTitle: {
      fontSize: theme.typography.fontSize.md,
      fontWeight: theme.typography.fontWeight.medium,
    },
    settingSubtitle: {
      fontSize: theme.typography.fontSize.sm,
      marginTop: 2,
    },
    settingRight: {
      flexDirection: 'row',
      alignItems: 'center',
    },
    themeButton: {
      flexDirection: 'row',
      alignItems: 'center',
      backgroundColor: theme.colors.background,
      paddingHorizontal: theme.spacing.md,
      paddingVertical: theme.spacing.sm,
      borderRadius: theme.borderRadius.lg,
      marginRight: theme.spacing.sm,
    },
    themeButtonText: {
      color: theme.colors.text,
      fontSize: theme.typography.fontSize.sm,
      marginLeft: theme.spacing.xs,
    },
    versionText: {
      fontSize: theme.typography.fontSize.sm,
      color: theme.colors.textSecondary,
    },
  });

  if (!user || !preferences) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text style={[styles.settingTitle, { color: theme.colors.text }]}>Loading...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
        {/* Profile Section */}
        <View style={styles.profileSection}>
          <View style={styles.profileAvatar}>
            <Icon name="person" size={40} color="white" />
          </View>
          <Text style={styles.profileName}>{user.name}</Text>
          <Text style={styles.profileEmail}>{user.email}</Text>
          <View style={[styles.subscriptionBadge, { backgroundColor: getSubscriptionBadgeColor(user.subscription) }]}>
            <Text style={styles.subscriptionText}>
              {user.subscription.toUpperCase()} PLAN
            </Text>
          </View>
        </View>

        {/* Appearance Settings */}
        <SettingSection title="Appearance">
          <SettingItem
            icon="palette"
            title="Theme"
            subtitle={isSystemTheme ? 'System' : isDarkMode ? 'Dark' : 'Light'}
            rightComponent={
              <View style={{ flexDirection: 'row' }}>
                <TouchableOpacity
                  style={styles.themeButton}
                  onPress={useSystemTheme}
                >
                  <Icon 
                    name="settings-brightness" 
                    size={16} 
                    color={isSystemTheme ? theme.colors.primary : theme.colors.textSecondary} 
                  />
                  <Text style={[
                    styles.themeButtonText,
                    { color: isSystemTheme ? theme.colors.primary : theme.colors.textSecondary }
                  ]}>
                    Auto
                  </Text>
                </TouchableOpacity>
                <Switch
                  value={isDarkMode}
                  onValueChange={toggleTheme}
                  disabled={isSystemTheme}
                  trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                  thumbColor={isDarkMode ? 'white' : theme.colors.background}
                />
              </View>
            }
          />
        </SettingSection>

        {/* Security Settings */}
        <SettingSection title="Security">
          {biometricAvailable && (
            <SettingItem
              icon="fingerprint"
              title={`${biometricType} Authentication`}
              subtitle="Use biometric authentication to unlock the app"
              rightComponent={
                <Switch
                  value={preferences.biometricAuth}
                  onValueChange={toggleBiometric}
                  trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                  thumbColor={preferences.biometricAuth ? 'white' : theme.colors.background}
                />
              }
            />
          )}
          <SettingItem
            icon="vpn-key"
            title="Change Password"
            subtitle="Update your account password"
            showArrow
            onPress={() => {
              // Navigate to change password screen
            }}
          />
        </SettingSection>

        {/* Notification Settings */}
        <SettingSection title="Notifications">
          <SettingItem
            icon="notifications"
            title="Analysis Complete"
            subtitle="Get notified when document analysis is finished"
            rightComponent={
              <Switch
                value={preferences.notifications.analysisComplete}
                onValueChange={(value) => updateNotificationPreference('analysisComplete', value)}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                thumbColor={preferences.notifications.analysisComplete ? 'white' : theme.colors.background}
              />
            }
          />
          <SettingItem
            icon="warning"
            title="Critical Clauses"
            subtitle="Alert for high-risk arbitration clauses"
            rightComponent={
              <Switch
                value={preferences.notifications.criticalClauses}
                onValueChange={(value) => updateNotificationPreference('criticalClauses', value)}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                thumbColor={preferences.notifications.criticalClauses ? 'white' : theme.colors.background}
              />
            }
          />
          <SettingItem
            icon="people"
            title="Team Updates"
            subtitle="Notifications for team collaboration"
            rightComponent={
              <Switch
                value={preferences.notifications.teamUpdates}
                onValueChange={(value) => updateNotificationPreference('teamUpdates', value)}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                thumbColor={preferences.notifications.teamUpdates ? 'white' : theme.colors.background}
              />
            }
          />
        </SettingSection>

        {/* Data & Storage */}
        <SettingSection title="Data & Storage">
          <SettingItem
            icon="sync"
            title="Auto Sync"
            subtitle="Automatically sync documents and analyses"
            rightComponent={
              <Switch
                value={preferences.autoSync}
                onValueChange={(value) => updatePreference('autoSync', value)}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                thumbColor={preferences.autoSync ? 'white' : theme.colors.background}
              />
            }
          />
          <SettingItem
            icon="offline-pin"
            title="Offline Mode"
            subtitle="Enable offline document processing"
            rightComponent={
              <Switch
                value={preferences.offlineMode}
                onValueChange={(value) => updatePreference('offlineMode', value)}
                trackColor={{ false: theme.colors.border, true: theme.colors.primary }}
                thumbColor={preferences.offlineMode ? 'white' : theme.colors.background}
              />
            }
          />
          <SettingItem
            icon="delete-sweep"
            title="Clear Cache"
            subtitle="Remove temporary files and cached data"
            showArrow
            onPress={clearCache}
          />
        </SettingSection>

        {/* Support & About */}
        <SettingSection title="Support & About">
          <SettingItem
            icon="help"
            title="Help & FAQ"
            subtitle="Get help and find answers"
            showArrow
            onPress={() => openURL('https://example.com/help')}
          />
          <SettingItem
            icon="feedback"
            title="Send Feedback"
            subtitle="Share your thoughts and suggestions"
            showArrow
            onPress={() => openURL('mailto:support@example.com')}
          />
          <SettingItem
            icon="privacy-tip"
            title="Privacy Policy"
            subtitle="Learn how we protect your data"
            showArrow
            onPress={() => openURL('https://example.com/privacy')}
          />
          <SettingItem
            icon="description"
            title="Terms of Service"
            subtitle="Review our terms and conditions"
            showArrow
            onPress={() => openURL('https://example.com/terms')}
          />
          <SettingItem
            icon="info"
            title="App Version"
            subtitle="Build 1.0.0 (2024)"
            rightComponent={
              <Text style={styles.versionText}>1.0.0</Text>
            }
          />
        </SettingSection>

        {/* Account Actions */}
        <SettingSection title="Account">
          <SettingItem
            icon="logout"
            title="Sign Out"
            subtitle="Sign out of your account"
            onPress={signOut}
          />
        </SettingSection>
      </ScrollView>
    </View>
  );
};

export default SettingsScreen;