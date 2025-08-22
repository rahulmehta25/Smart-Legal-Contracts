import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { useTheme } from '@hooks/useTheme';
import { BottomTabParamList, RootStackParamList } from '@types/index';

// Screens
import HomeScreen from '@screens/HomeScreen';
import ScannerScreen from '@screens/ScannerScreen';
import AnalysisScreen from '@screens/AnalysisScreen';
import HistoryScreen from '@screens/HistoryScreen';
import SettingsScreen from '@screens/SettingsScreen';
import DocumentDetailsScreen from '@screens/DocumentDetailsScreen';
import BiometricSetupScreen from '@screens/BiometricSetupScreen';
import OnboardingFlowScreen from '@screens/OnboardingFlowScreen';

const Tab = createBottomTabNavigator<BottomTabParamList>();
const Stack = createStackNavigator<RootStackParamList>();

const TabNavigator: React.FC = () => {
  const { theme } = useTheme();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName: string;

          switch (route.name) {
            case 'HomeTab':
              iconName = 'dashboard';
              break;
            case 'ScannerTab':
              iconName = 'document-scanner';
              break;
            case 'HistoryTab':
              iconName = 'history';
              break;
            case 'SettingsTab':
              iconName = 'settings';
              break;
            default:
              iconName = 'help';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.textSecondary,
        tabBarStyle: {
          backgroundColor: theme.colors.surface,
          borderTopColor: theme.colors.border,
          paddingBottom: 8,
          paddingTop: 8,
          height: 70,
        },
        headerStyle: {
          backgroundColor: theme.colors.surface,
          shadowColor: theme.colors.shadow,
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: 0.1,
          shadowRadius: 4,
          elevation: 4,
        },
        headerTintColor: theme.colors.text,
        headerTitleStyle: {
          fontWeight: theme.typography.fontWeight.medium,
          fontSize: theme.typography.fontSize.lg,
        },
      })}
    >
      <Tab.Screen 
        name="HomeTab" 
        component={HomeScreen} 
        options={{ 
          title: 'Dashboard',
          tabBarLabel: 'Home',
        }} 
      />
      <Tab.Screen 
        name="ScannerTab" 
        component={ScannerScreen} 
        options={{ 
          title: 'Document Scanner',
          tabBarLabel: 'Scan',
        }} 
      />
      <Tab.Screen 
        name="HistoryTab" 
        component={HistoryScreen} 
        options={{ 
          title: 'Document History',
          tabBarLabel: 'History',
        }} 
      />
      <Tab.Screen 
        name="SettingsTab" 
        component={SettingsScreen} 
        options={{ 
          title: 'Settings',
          tabBarLabel: 'Settings',
        }} 
      />
    </Tab.Navigator>
  );
};

const AppNavigator: React.FC = () => {
  const { theme } = useTheme();

  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerStyle: {
            backgroundColor: theme.colors.surface,
            shadowColor: theme.colors.shadow,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 4,
            elevation: 4,
          },
          headerTintColor: theme.colors.text,
          headerTitleStyle: {
            fontWeight: theme.typography.fontWeight.medium,
            fontSize: theme.typography.fontSize.lg,
          },
          cardStyle: {
            backgroundColor: theme.colors.background,
          },
        }}
      >
        <Stack.Screen 
          name="Home" 
          component={TabNavigator} 
          options={{ headerShown: false }} 
        />
        <Stack.Screen 
          name="Analysis" 
          component={AnalysisScreen}
          options={{ 
            title: 'Analysis Results',
            headerBackTitleVisible: false,
          }} 
        />
        <Stack.Screen 
          name="DocumentDetails" 
          component={DocumentDetailsScreen}
          options={{ 
            title: 'Document Details',
            headerBackTitleVisible: false,
          }} 
        />
        <Stack.Screen 
          name="BiometricSetup" 
          component={BiometricSetupScreen}
          options={{ 
            title: 'Biometric Setup',
            headerBackTitleVisible: false,
          }} 
        />
        <Stack.Screen 
          name="OnboardingFlow" 
          component={OnboardingFlowScreen}
          options={{ 
            headerShown: false,
            gestureEnabled: false,
          }} 
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;