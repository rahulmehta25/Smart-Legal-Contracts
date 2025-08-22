import AsyncStorage from '@react-native-async-storage/async-storage';
import { User, UserPreferences, SubscriptionType } from '@types/index';
import { generateUUID } from '@utils/helpers';

class UserService {
  private currentUser: User | null = null;
  private readonly STORAGE_KEYS = {
    USER: 'current_user',
    PREFERENCES: 'user_preferences',
    AUTH_TOKEN: 'auth_token',
  };

  async getCurrentUser(): Promise<User | null> {
    try {
      if (this.currentUser) {
        return this.currentUser;
      }

      const userData = await AsyncStorage.getItem(this.STORAGE_KEYS.USER);
      if (userData) {
        this.currentUser = JSON.parse(userData);
        return this.currentUser;
      }

      // Create a default user if none exists (for demo purposes)
      const defaultUser = await this.createDefaultUser();
      await this.saveUser(defaultUser);
      return defaultUser;
    } catch (error) {
      console.error('Error getting current user:', error);
      return null;
    }
  }

  async saveUser(user: User): Promise<void> {
    try {
      await AsyncStorage.setItem(this.STORAGE_KEYS.USER, JSON.stringify(user));
      this.currentUser = user;
    } catch (error) {
      console.error('Error saving user:', error);
      throw new Error('Failed to save user data');
    }
  }

  async updateUser(updates: Partial<User>): Promise<User> {
    try {
      const currentUser = await this.getCurrentUser();
      if (!currentUser) {
        throw new Error('No user found to update');
      }

      const updatedUser: User = {
        ...currentUser,
        ...updates,
      };

      await this.saveUser(updatedUser);
      return updatedUser;
    } catch (error) {
      console.error('Error updating user:', error);
      throw new Error('Failed to update user');
    }
  }

  async updateUserPreferences(preferences: UserPreferences): Promise<void> {
    try {
      const currentUser = await this.getCurrentUser();
      if (!currentUser) {
        throw new Error('No user found');
      }

      const updatedUser = {
        ...currentUser,
        preferences,
      };

      await this.saveUser(updatedUser);
      await AsyncStorage.setItem(this.STORAGE_KEYS.PREFERENCES, JSON.stringify(preferences));
    } catch (error) {
      console.error('Error updating user preferences:', error);
      throw new Error('Failed to update preferences');
    }
  }

  async getUserPreferences(): Promise<UserPreferences> {
    try {
      const currentUser = await this.getCurrentUser();
      if (currentUser?.preferences) {
        return currentUser.preferences;
      }

      // Return default preferences
      return this.getDefaultPreferences();
    } catch (error) {
      console.error('Error getting user preferences:', error);
      return this.getDefaultPreferences();
    }
  }

  async signIn(email: string, password: string): Promise<{
    success: boolean;
    user?: User;
    token?: string;
    error?: string;
  }> {
    try {
      // This would typically make an API call to your authentication service
      // For demo purposes, we'll simulate authentication
      
      if (email && password) {
        const user: User = {
          id: generateUUID(),
          email,
          name: this.getNameFromEmail(email),
          preferences: this.getDefaultPreferences(),
          subscription: SubscriptionType.FREE,
        };

        const token = `mock_token_${Date.now()}`;
        
        await this.saveUser(user);
        await AsyncStorage.setItem(this.STORAGE_KEYS.AUTH_TOKEN, token);
        
        return {
          success: true,
          user,
          token,
        };
      } else {
        return {
          success: false,
          error: 'Invalid credentials',
        };
      }
    } catch (error) {
      console.error('Error signing in:', error);
      return {
        success: false,
        error: 'Sign in failed',
      };
    }
  }

  async signUp(email: string, password: string, name: string): Promise<{
    success: boolean;
    user?: User;
    token?: string;
    error?: string;
  }> {
    try {
      // This would typically make an API call to your registration service
      
      const user: User = {
        id: generateUUID(),
        email,
        name,
        preferences: this.getDefaultPreferences(),
        subscription: SubscriptionType.FREE,
      };

      const token = `mock_token_${Date.now()}`;
      
      await this.saveUser(user);
      await AsyncStorage.setItem(this.STORAGE_KEYS.AUTH_TOKEN, token);
      
      return {
        success: true,
        user,
        token,
      };
    } catch (error) {
      console.error('Error signing up:', error);
      return {
        success: false,
        error: 'Sign up failed',
      };
    }
  }

  async signOut(): Promise<void> {
    try {
      await AsyncStorage.multiRemove([
        this.STORAGE_KEYS.USER,
        this.STORAGE_KEYS.AUTH_TOKEN,
        this.STORAGE_KEYS.PREFERENCES,
      ]);
      
      this.currentUser = null;
    } catch (error) {
      console.error('Error signing out:', error);
      throw new Error('Failed to sign out');
    }
  }

  async getAuthToken(): Promise<string | null> {
    try {
      return await AsyncStorage.getItem(this.STORAGE_KEYS.AUTH_TOKEN);
    } catch (error) {
      console.error('Error getting auth token:', error);
      return null;
    }
  }

  async isAuthenticated(): Promise<boolean> {
    try {
      const token = await this.getAuthToken();
      const user = await this.getCurrentUser();
      return !!(token && user);
    } catch (error) {
      console.error('Error checking authentication:', error);
      return false;
    }
  }

  async refreshAuthToken(): Promise<string | null> {
    try {
      // This would typically make an API call to refresh the token
      const newToken = `refreshed_token_${Date.now()}`;
      await AsyncStorage.setItem(this.STORAGE_KEYS.AUTH_TOKEN, newToken);
      return newToken;
    } catch (error) {
      console.error('Error refreshing auth token:', error);
      return null;
    }
  }

  async updateSubscription(subscription: SubscriptionType): Promise<void> {
    try {
      await this.updateUser({ subscription });
    } catch (error) {
      console.error('Error updating subscription:', error);
      throw new Error('Failed to update subscription');
    }
  }

  async updateAvatar(avatarUri: string): Promise<void> {
    try {
      await this.updateUser({ avatar: avatarUri });
    } catch (error) {
      console.error('Error updating avatar:', error);
      throw new Error('Failed to update avatar');
    }
  }

  async deleteAccount(): Promise<boolean> {
    try {
      // This would typically make an API call to delete the account
      await this.signOut();
      return true;
    } catch (error) {
      console.error('Error deleting account:', error);
      return false;
    }
  }

  async changePassword(currentPassword: string, newPassword: string): Promise<{
    success: boolean;
    error?: string;
  }> {
    try {
      // This would typically make an API call to change the password
      // For demo purposes, we'll just simulate success
      return { success: true };
    } catch (error) {
      console.error('Error changing password:', error);
      return {
        success: false,
        error: 'Failed to change password',
      };
    }
  }

  async resetPassword(email: string): Promise<{
    success: boolean;
    error?: string;
  }> {
    try {
      // This would typically make an API call to initiate password reset
      return { success: true };
    } catch (error) {
      console.error('Error resetting password:', error);
      return {
        success: false,
        error: 'Failed to reset password',
      };
    }
  }

  async getUserProfile(): Promise<{
    documentsAnalyzed: number;
    highRiskDocuments: number;
    totalScanTime: number;
    averageConfidence: number;
    memberSince: Date;
  }> {
    try {
      const user = await this.getCurrentUser();
      if (!user) {
        throw new Error('No user found');
      }

      // This would typically fetch actual analytics data
      return {
        documentsAnalyzed: 42,
        highRiskDocuments: 7,
        totalScanTime: 1234567, // milliseconds
        averageConfidence: 0.89,
        memberSince: new Date('2024-01-01'),
      };
    } catch (error) {
      console.error('Error getting user profile:', error);
      throw new Error('Failed to get user profile');
    }
  }

  private async createDefaultUser(): Promise<User> {
    return {
      id: generateUUID(),
      email: 'user@example.com',
      name: 'Demo User',
      preferences: this.getDefaultPreferences(),
      subscription: SubscriptionType.FREE,
    };
  }

  private getDefaultPreferences(): UserPreferences {
    return {
      darkMode: false,
      biometricAuth: false,
      notifications: {
        analysisComplete: true,
        criticalClauses: true,
        teamUpdates: false,
        weeklyReports: true,
      },
      language: 'en',
      autoSync: true,
      offlineMode: false,
    };
  }

  private getNameFromEmail(email: string): string {
    const username = email.split('@')[0];
    return username.charAt(0).toUpperCase() + username.slice(1);
  }

  async exportUserData(): Promise<string> {
    try {
      const user = await this.getCurrentUser();
      const preferences = await this.getUserPreferences();
      const profile = await this.getUserProfile();

      const userData = {
        user,
        preferences,
        profile,
        exportDate: new Date().toISOString(),
      };

      return JSON.stringify(userData, null, 2);
    } catch (error) {
      console.error('Error exporting user data:', error);
      throw new Error('Failed to export user data');
    }
  }

  async clearUserData(): Promise<void> {
    try {
      await AsyncStorage.multiRemove([
        this.STORAGE_KEYS.USER,
        this.STORAGE_KEYS.PREFERENCES,
        this.STORAGE_KEYS.AUTH_TOKEN,
      ]);
      
      this.currentUser = null;
    } catch (error) {
      console.error('Error clearing user data:', error);
      throw new Error('Failed to clear user data');
    }
  }
}

export const userService = new UserService();