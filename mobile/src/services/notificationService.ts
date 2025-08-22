import PushNotification from 'react-native-push-notification';
import messaging from '@react-native-firebase/messaging';
import { Platform, PermissionsAndroid, Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NotificationSettings } from '@types/index';

interface NotificationPayload {
  title: string;
  message: string;
  data?: any;
  type: 'analysis_complete' | 'critical_clause' | 'team_update' | 'weekly_report';
}

class NotificationService {
  private fcmToken: string | null = null;
  private channelCreated = false;

  constructor() {
    this.initializeNotifications();
  }

  private async initializeNotifications(): Promise<void> {
    try {
      // Configure PushNotification
      PushNotification.configure({
        onRegister: (token) => {
          console.log('FCM Token:', token);
          this.fcmToken = token.token;
          this.saveFCMToken(token.token);
        },

        onNotification: (notification) => {
          console.log('Notification received:', notification);
          this.handleNotification(notification);
        },

        onAction: (notification) => {
          console.log('Notification action:', notification);
          this.handleNotificationAction(notification);
        },

        onRegistrationError: (err) => {
          console.error('Notification registration error:', err);
        },

        permissions: {
          alert: true,
          badge: true,
          sound: true,
        },

        popInitialNotification: true,
        requestPermissions: false, // We'll request manually
      });

      // Create notification channel for Android
      if (Platform.OS === 'android') {
        this.createNotificationChannel();
      }

      // Initialize Firebase messaging
      await this.initializeFirebaseMessaging();
    } catch (error) {
      console.error('Error initializing notifications:', error);
    }
  }

  private async initializeFirebaseMessaging(): Promise<void> {
    try {
      // Request permission
      const authStatus = await messaging().requestPermission();
      const enabled =
        authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
        authStatus === messaging.AuthorizationStatus.PROVISIONAL;

      if (enabled) {
        console.log('Notification permission granted');
        
        // Get FCM token
        const token = await messaging().getToken();
        this.fcmToken = token;
        await this.saveFCMToken(token);

        // Listen for token refresh
        messaging().onTokenRefresh(async (newToken) => {
          console.log('FCM token refreshed:', newToken);
          this.fcmToken = newToken;
          await this.saveFCMToken(newToken);
        });

        // Handle foreground messages
        messaging().onMessage(async (remoteMessage) => {
          console.log('Foreground message:', remoteMessage);
          this.showLocalNotification({
            title: remoteMessage.notification?.title || 'Notification',
            message: remoteMessage.notification?.body || '',
            data: remoteMessage.data,
            type: (remoteMessage.data?.type as any) || 'analysis_complete',
          });
        });

        // Handle background/quit state messages
        messaging().setBackgroundMessageHandler(async (remoteMessage) => {
          console.log('Background message:', remoteMessage);
          // Handle background message processing
        });
      } else {
        console.log('Notification permission denied');
      }
    } catch (error) {
      console.error('Error initializing Firebase messaging:', error);
    }
  }

  private createNotificationChannel(): void {
    if (this.channelCreated) return;

    PushNotification.createChannel(
      {
        channelId: 'arbitration-detector',
        channelName: 'Arbitration Detector',
        channelDescription: 'Notifications for document analysis and alerts',
        playSound: true,
        soundName: 'default',
        importance: 4,
        vibrate: true,
      },
      (created) => {
        console.log('Notification channel created:', created);
        this.channelCreated = true;
      }
    );
  }

  async requestPermissions(): Promise<boolean> {
    try {
      if (Platform.OS === 'ios') {
        return new Promise((resolve) => {
          PushNotification.requestPermissions()
            .then((permissions) => {
              console.log('iOS notification permissions:', permissions);
              resolve(permissions.alert && permissions.badge && permissions.sound);
            })
            .catch((error) => {
              console.error('Error requesting iOS permissions:', error);
              resolve(false);
            });
        });
      } else {
        // Android permissions
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.POST_NOTIFICATIONS
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      }
    } catch (error) {
      console.error('Error requesting notification permissions:', error);
      return false;
    }
  }

  async showLocalNotification(payload: NotificationPayload): Promise<void> {
    try {
      const settings = await this.getNotificationSettings();
      
      // Check if this type of notification is enabled
      if (!this.isNotificationTypeEnabled(payload.type, settings)) {
        return;
      }

      PushNotification.localNotification({
        channelId: 'arbitration-detector',
        title: payload.title,
        message: payload.message,
        playSound: true,
        soundName: 'default',
        number: 1,
        vibrate: true,
        vibration: 300,
        userInfo: payload.data,
        actions: this.getNotificationActions(payload.type),
      });
    } catch (error) {
      console.error('Error showing local notification:', error);
    }
  }

  async scheduleNotification(
    payload: NotificationPayload,
    date: Date
  ): Promise<void> {
    try {
      const settings = await this.getNotificationSettings();
      
      if (!this.isNotificationTypeEnabled(payload.type, settings)) {
        return;
      }

      PushNotification.localNotificationSchedule({
        channelId: 'arbitration-detector',
        title: payload.title,
        message: payload.message,
        date,
        playSound: true,
        soundName: 'default',
        userInfo: payload.data,
        actions: this.getNotificationActions(payload.type),
      });
    } catch (error) {
      console.error('Error scheduling notification:', error);
    }
  }

  async sendAnalysisCompleteNotification(
    documentName: string,
    hasArbitrationClause: boolean,
    riskLevel: string
  ): Promise<void> {
    const title = 'Analysis Complete';
    const message = hasArbitrationClause
      ? `⚠️ Arbitration clause detected in "${documentName}" (${riskLevel} risk)`
      : `✅ No arbitration clause found in "${documentName}"`;

    await this.showLocalNotification({
      title,
      message,
      type: 'analysis_complete',
      data: { documentName, hasArbitrationClause, riskLevel },
    });
  }

  async sendCriticalClauseAlert(
    documentName: string,
    clauseType: string
  ): Promise<void> {
    await this.showLocalNotification({
      title: 'Critical Clause Detected',
      message: `🚨 High-risk ${clauseType} found in "${documentName}"`,
      type: 'critical_clause',
      data: { documentName, clauseType },
    });
  }

  async sendTeamUpdateNotification(
    updateType: string,
    message: string
  ): Promise<void> {
    await this.showLocalNotification({
      title: 'Team Update',
      message,
      type: 'team_update',
      data: { updateType },
    });
  }

  async sendWeeklyReport(
    documentsAnalyzed: number,
    highRiskDocuments: number
  ): Promise<void> {
    const message = `This week: ${documentsAnalyzed} documents analyzed, ${highRiskDocuments} high-risk clauses found`;
    
    await this.showLocalNotification({
      title: 'Weekly Report',
      message,
      type: 'weekly_report',
      data: { documentsAnalyzed, highRiskDocuments },
    });
  }

  async scheduleWeeklyReports(): Promise<void> {
    try {
      // Cancel existing weekly notifications
      PushNotification.cancelAllLocalNotifications();

      // Schedule weekly report for every Sunday at 9 AM
      const now = new Date();
      const nextSunday = new Date(now);
      nextSunday.setDate(now.getDate() + (7 - now.getDay()));
      nextSunday.setHours(9, 0, 0, 0);

      // Get weekly stats (this would be implemented to fetch actual data)
      const stats = await this.getWeeklyStats();

      await this.scheduleNotification(
        {
          title: 'Weekly Report',
          message: `${stats.documentsAnalyzed} documents analyzed this week`,
          type: 'weekly_report',
          data: stats,
        },
        nextSunday
      );
    } catch (error) {
      console.error('Error scheduling weekly reports:', error);
    }
  }

  private async getWeeklyStats(): Promise<{
    documentsAnalyzed: number;
    highRiskDocuments: number;
  }> {
    // This would integrate with your analytics service
    // For now, return mock data
    return {
      documentsAnalyzed: 5,
      highRiskDocuments: 2,
    };
  }

  private getNotificationActions(type: string): string[] {
    switch (type) {
      case 'analysis_complete':
        return ['View Results', 'Dismiss'];
      case 'critical_clause':
        return ['View Details', 'Get Advice', 'Dismiss'];
      case 'team_update':
        return ['View Update', 'Dismiss'];
      case 'weekly_report':
        return ['View Report', 'Dismiss'];
      default:
        return ['Dismiss'];
    }
  }

  private handleNotification(notification: any): void {
    try {
      // Handle notification tap/interaction
      if (notification.userInteraction) {
        // User tapped the notification
        this.navigateToRelevantScreen(notification);
      }
    } catch (error) {
      console.error('Error handling notification:', error);
    }
  }

  private handleNotificationAction(notification: any): void {
    try {
      const action = notification.action;
      const data = notification.userInfo || notification.data;

      switch (action) {
        case 'View Results':
        case 'View Details':
          this.navigateToAnalysisScreen(data);
          break;
        case 'Get Advice':
          this.navigateToAdviceScreen(data);
          break;
        case 'View Update':
          this.navigateToTeamScreen(data);
          break;
        case 'View Report':
          this.navigateToReportScreen(data);
          break;
        default:
          // Dismiss or unknown action
          break;
      }
    } catch (error) {
      console.error('Error handling notification action:', error);
    }
  }

  private navigateToRelevantScreen(notification: any): void {
    // This would integrate with your navigation system
    const data = notification.userInfo || notification.data;
    const type = data?.type;

    switch (type) {
      case 'analysis_complete':
      case 'critical_clause':
        this.navigateToAnalysisScreen(data);
        break;
      case 'team_update':
        this.navigateToTeamScreen(data);
        break;
      case 'weekly_report':
        this.navigateToReportScreen(data);
        break;
      default:
        // Navigate to home
        break;
    }
  }

  private navigateToAnalysisScreen(data: any): void {
    // Implementation would use your navigation system
    console.log('Navigate to analysis screen with data:', data);
  }

  private navigateToAdviceScreen(data: any): void {
    console.log('Navigate to advice screen with data:', data);
  }

  private navigateToTeamScreen(data: any): void {
    console.log('Navigate to team screen with data:', data);
  }

  private navigateToReportScreen(data: any): void {
    console.log('Navigate to report screen with data:', data);
  }

  private async getNotificationSettings(): Promise<NotificationSettings> {
    try {
      const settings = await AsyncStorage.getItem('notification_settings');
      if (settings) {
        return JSON.parse(settings);
      }
    } catch (error) {
      console.error('Error getting notification settings:', error);
    }

    // Default settings
    return {
      analysisComplete: true,
      criticalClauses: true,
      teamUpdates: true,
      weeklyReports: true,
    };
  }

  private isNotificationTypeEnabled(
    type: NotificationPayload['type'],
    settings: NotificationSettings
  ): boolean {
    switch (type) {
      case 'analysis_complete':
        return settings.analysisComplete;
      case 'critical_clause':
        return settings.criticalClauses;
      case 'team_update':
        return settings.teamUpdates;
      case 'weekly_report':
        return settings.weeklyReports;
      default:
        return true;
    }
  }

  private async saveFCMToken(token: string): Promise<void> {
    try {
      await AsyncStorage.setItem('fcm_token', token);
      // Also send to your backend for server-side notifications
      // await this.sendTokenToServer(token);
    } catch (error) {
      console.error('Error saving FCM token:', error);
    }
  }

  async getFCMToken(): Promise<string | null> {
    try {
      if (this.fcmToken) {
        return this.fcmToken;
      }
      
      return await AsyncStorage.getItem('fcm_token');
    } catch (error) {
      console.error('Error getting FCM token:', error);
      return null;
    }
  }

  async clearAllNotifications(): Promise<void> {
    try {
      PushNotification.cancelAllLocalNotifications();
      PushNotification.removeAllDeliveredNotifications();
    } catch (error) {
      console.error('Error clearing notifications:', error);
    }
  }

  async getNotificationHistory(): Promise<any[]> {
    try {
      // This would return notification history from storage
      // For now, return empty array
      return [];
    } catch (error) {
      console.error('Error getting notification history:', error);
      return [];
    }
  }

  async updateNotificationSettings(settings: NotificationSettings): Promise<void> {
    try {
      await AsyncStorage.setItem('notification_settings', JSON.stringify(settings));
      
      // If weekly reports are disabled, cancel scheduled reports
      if (!settings.weeklyReports) {
        PushNotification.cancelAllLocalNotifications();
      } else {
        await this.scheduleWeeklyReports();
      }
    } catch (error) {
      console.error('Error updating notification settings:', error);
      throw new Error('Failed to update notification settings');
    }
  }
}

export const notificationService = new NotificationService();