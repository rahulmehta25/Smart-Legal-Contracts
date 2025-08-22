import React, { useEffect, useState } from 'react';
import { StatusBar, Alert, Platform } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { QueryClient, QueryClientProvider } from 'react-query';
import AppNavigator from '@navigation/AppNavigator';
import { useTheme } from '@hooks/useTheme';
import { biometricService } from '@services/biometricService';
import { notificationService } from '@services/notificationService';
import { userService } from '@services/userService';
import SplashScreen from 'react-native-splash-screen';

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

const App: React.FC = () => {
  const { theme, isDarkMode } = useTheme();
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Hide splash screen
      if (Platform.OS === 'ios') {
        SplashScreen.hide();
      }

      // Check if user has completed onboarding
      const hasCompletedOnboarding = await checkOnboardingStatus();
      
      if (!hasCompletedOnboarding) {
        setShowOnboarding(true);
        setIsLoading(false);
        return;
      }

      // Initialize services
      await initializeServices();

      // Check authentication status
      const authenticated = await userService.isAuthenticated();
      setIsAuthenticated(authenticated);

      if (authenticated) {
        // Check if biometric authentication is enabled
        const biometricEnabled = await biometricService.isBiometricEnabled();
        
        if (biometricEnabled) {
          const biometricResult = await biometricService.authenticateForAppAccess();
          
          if (!biometricResult) {
            Alert.alert(
              'Authentication Required',
              'Biometric authentication failed. Please try again.',
              [
                { text: 'Retry', onPress: () => initializeApp() },
                { text: 'Exit App', onPress: () => {/* Exit app */} }
              ]
            );
            return;
          }
        }
      }

    } catch (error) {
      console.error('Error initializing app:', error);
      Alert.alert(
        'Initialization Error',
        'Failed to initialize the app. Please restart and try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const checkOnboardingStatus = async (): Promise<boolean> => {
    try {
      // Check if onboarding has been completed
      // This would typically be stored in AsyncStorage
      return true; // For demo, assume onboarding is complete
    } catch (error) {
      console.error('Error checking onboarding status:', error);
      return false;
    }
  };

  const initializeServices = async () => {
    try {
      // Initialize notification service
      await notificationService.requestPermissions();
      
      // Schedule weekly reports if enabled
      const userPreferences = await userService.getUserPreferences();
      if (userPreferences.notifications.weeklyReports) {
        await notificationService.scheduleWeeklyReports();
      }

      console.log('Services initialized successfully');
    } catch (error) {
      console.error('Error initializing services:', error);
    }
  };

  if (isLoading) {
    // You could show a custom loading screen here
    return null;
  }

  return (
    <QueryClientProvider client={queryClient}>
      <GestureHandlerRootView style={{ flex: 1 }}>
        <SafeAreaProvider>
          <StatusBar
            barStyle={isDarkMode ? 'light-content' : 'dark-content'}
            backgroundColor={theme.colors.background}
            translucent={false}
          />
          <AppNavigator />
        </SafeAreaProvider>
      </GestureHandlerRootView>
    </QueryClientProvider>
  );
};

export default App;