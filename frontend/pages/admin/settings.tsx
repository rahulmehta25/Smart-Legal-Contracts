import React, { useState, useEffect } from 'react';
import { CogIcon, KeyIcon, BellIcon, ShieldCheckIcon, UserGroupIcon, BuildingOfficeIcon } from '@heroicons/react/24/outline';
import AdminLayout from '../../src/components/admin/AdminLayout';
import SettingsSection from '../../src/components/admin/SettingsSection';
import FeatureFlagsPanel from '../../src/components/admin/FeatureFlagsPanel';
import APIKeyManager from '../../src/components/admin/APIKeyManager';
import SystemAnnouncementManager from '../../src/components/admin/SystemAnnouncementManager';
import AuditLogViewer from '../../src/components/admin/AuditLogViewer';
import NotificationSettings from '../../src/components/admin/NotificationSettings';

interface SystemSettings {
  general: {
    siteName: string;
    siteDescription: string;
    supportEmail: string;
    maintenanceMode: boolean;
    maxFileSize: number;
    allowedFileTypes: string[];
  };
  security: {
    passwordMinLength: number;
    requireTwoFactor: boolean;
    sessionTimeout: number;
    maxLoginAttempts: number;
    ipWhitelist: string[];
  };
  notifications: {
    emailNotifications: boolean;
    slackWebhookUrl: string;
    discordWebhookUrl: string;
    alertThresholds: {
      errorRate: number;
      responseTime: number;
      diskUsage: number;
      memoryUsage: number;
    };
  };
  analytics: {
    enableAnalytics: boolean;
    dataRetentionDays: number;
    anonymizeUserData: boolean;
  };
}

const SettingsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState<SystemSettings>({
    general: {
      siteName: 'Arbitration Detection Platform',
      siteDescription: 'AI-powered legal document analysis',
      supportEmail: 'support@example.com',
      maintenanceMode: false,
      maxFileSize: 50,
      allowedFileTypes: ['pdf', 'docx', 'txt']
    },
    security: {
      passwordMinLength: 8,
      requireTwoFactor: false,
      sessionTimeout: 30,
      maxLoginAttempts: 5,
      ipWhitelist: []
    },
    notifications: {
      emailNotifications: true,
      slackWebhookUrl: '',
      discordWebhookUrl: '',
      alertThresholds: {
        errorRate: 5,
        responseTime: 2000,
        diskUsage: 80,
        memoryUsage: 85
      }
    },
    analytics: {
      enableAnalytics: true,
      dataRetentionDays: 365,
      anonymizeUserData: true
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setIsLoading(true);
    // Mock API call - replace with actual implementation
    setTimeout(() => {
      // Settings already initialized above
      setIsLoading(false);
    }, 500);
  };

  const saveSettings = async () => {
    setIsLoading(true);
    setSaveMessage(null);
    
    try {
      // Mock API call - replace with actual implementation
      await new Promise(resolve => setTimeout(resolve, 1000));
      setSaveMessage('Settings saved successfully');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (error) {
      setSaveMessage('Error saving settings');
    } finally {
      setIsLoading(false);
    }
  };

  const updateSettings = (section: keyof SystemSettings, updates: Partial<SystemSettings[keyof SystemSettings]>) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        ...updates
      }
    }));
  };

  const tabs = [
    { id: 'general', name: 'General', icon: CogIcon },
    { id: 'security', name: 'Security', icon: ShieldCheckIcon },
    { id: 'api-keys', name: 'API Keys', icon: KeyIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
    { id: 'features', name: 'Feature Flags', icon: UserGroupIcon },
    { id: 'announcements', name: 'Announcements', icon: BuildingOfficeIcon },
    { id: 'audit-logs', name: 'Audit Logs', icon: BuildingOfficeIcon },
  ];

  return (
    <AdminLayout>
      <div className="space-y-6" id="settings-page-container">
        {/* Header */}
        <div className="border-b border-gray-200 pb-4" id="settings-header">
          <div className="sm:flex sm:items-center sm:justify-between" id="settings-header-content">
            <div id="settings-title-section">
              <h1 className="text-2xl font-semibold text-gray-900" id="settings-title">
                System Settings
              </h1>
              <p className="mt-1 text-sm text-gray-500" id="settings-subtitle">
                Configure system behavior and administrative options
              </p>
            </div>
            <div className="mt-4 sm:mt-0" id="settings-actions">
              <button
                onClick={saveSettings}
                disabled={isLoading}
                className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                id="save-settings-btn"
              >
                {isLoading ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>
        </div>

        {/* Save Message */}
        {saveMessage && (
          <div className={`rounded-md p-4 ${
            saveMessage.includes('Error') 
              ? 'bg-red-50 border border-red-200' 
              : 'bg-green-50 border border-green-200'
          }`} id="save-message">
            <p className={`text-sm ${
              saveMessage.includes('Error') ? 'text-red-700' : 'text-green-700'
            }`} id="save-message-text">
              {saveMessage}
            </p>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6" id="settings-layout">
          {/* Tabs Navigation */}
          <div className="lg:w-64" id="settings-nav">
            <nav className="space-y-1" id="settings-nav-list">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md ${
                      activeTab === tab.id
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                    id={`${tab.id}-tab`}
                  >
                    <Icon className="mr-3 h-5 w-5" id={`${tab.id}-tab-icon`} />
                    {tab.name}
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Settings Content */}
          <div className="flex-1" id="settings-content">
            <div className="bg-white rounded-lg border border-gray-200" id="settings-panel">
              {activeTab === 'general' && (
                <SettingsSection title="General Settings" id="general-settings-section">
                  <div className="space-y-6" id="general-settings-content">
                    <div id="site-name-setting">
                      <label htmlFor="siteName" className="block text-sm font-medium text-gray-700">
                        Site Name
                      </label>
                      <input
                        type="text"
                        id="siteName"
                        value={settings.general.siteName}
                        onChange={(e) => updateSettings('general', { siteName: e.target.value })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div id="site-description-setting">
                      <label htmlFor="siteDescription" className="block text-sm font-medium text-gray-700">
                        Site Description
                      </label>
                      <textarea
                        id="siteDescription"
                        rows={3}
                        value={settings.general.siteDescription}
                        onChange={(e) => updateSettings('general', { siteDescription: e.target.value })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div id="support-email-setting">
                      <label htmlFor="supportEmail" className="block text-sm font-medium text-gray-700">
                        Support Email
                      </label>
                      <input
                        type="email"
                        id="supportEmail"
                        value={settings.general.supportEmail}
                        onChange={(e) => updateSettings('general', { supportEmail: e.target.value })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div id="maintenance-mode-setting">
                      <div className="flex items-center justify-between">
                        <div>
                          <label htmlFor="maintenanceMode" className="text-sm font-medium text-gray-700">
                            Maintenance Mode
                          </label>
                          <p className="text-sm text-gray-500">
                            Temporarily disable the application for maintenance
                          </p>
                        </div>
                        <input
                          type="checkbox"
                          id="maintenanceMode"
                          checked={settings.general.maintenanceMode}
                          onChange={(e) => updateSettings('general', { maintenanceMode: e.target.checked })}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                      </div>
                    </div>

                    <div id="max-file-size-setting">
                      <label htmlFor="maxFileSize" className="block text-sm font-medium text-gray-700">
                        Maximum File Size (MB)
                      </label>
                      <input
                        type="number"
                        id="maxFileSize"
                        min="1"
                        max="500"
                        value={settings.general.maxFileSize}
                        onChange={(e) => updateSettings('general', { maxFileSize: parseInt(e.target.value) })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                </SettingsSection>
              )}

              {activeTab === 'security' && (
                <SettingsSection title="Security Settings" id="security-settings-section">
                  <div className="space-y-6" id="security-settings-content">
                    <div id="password-min-length-setting">
                      <label htmlFor="passwordMinLength" className="block text-sm font-medium text-gray-700">
                        Minimum Password Length
                      </label>
                      <input
                        type="number"
                        id="passwordMinLength"
                        min="6"
                        max="20"
                        value={settings.security.passwordMinLength}
                        onChange={(e) => updateSettings('security', { passwordMinLength: parseInt(e.target.value) })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div id="two-factor-setting">
                      <div className="flex items-center justify-between">
                        <div>
                          <label htmlFor="requireTwoFactor" className="text-sm font-medium text-gray-700">
                            Require Two-Factor Authentication
                          </label>
                          <p className="text-sm text-gray-500">
                            Force all users to enable 2FA
                          </p>
                        </div>
                        <input
                          type="checkbox"
                          id="requireTwoFactor"
                          checked={settings.security.requireTwoFactor}
                          onChange={(e) => updateSettings('security', { requireTwoFactor: e.target.checked })}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                      </div>
                    </div>

                    <div id="session-timeout-setting">
                      <label htmlFor="sessionTimeout" className="block text-sm font-medium text-gray-700">
                        Session Timeout (minutes)
                      </label>
                      <input
                        type="number"
                        id="sessionTimeout"
                        min="5"
                        max="1440"
                        value={settings.security.sessionTimeout}
                        onChange={(e) => updateSettings('security', { sessionTimeout: parseInt(e.target.value) })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>

                    <div id="max-login-attempts-setting">
                      <label htmlFor="maxLoginAttempts" className="block text-sm font-medium text-gray-700">
                        Maximum Login Attempts
                      </label>
                      <input
                        type="number"
                        id="maxLoginAttempts"
                        min="3"
                        max="10"
                        value={settings.security.maxLoginAttempts}
                        onChange={(e) => updateSettings('security', { maxLoginAttempts: parseInt(e.target.value) })}
                        className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                      />
                    </div>
                  </div>
                </SettingsSection>
              )}

              {activeTab === 'api-keys' && (
                <div id="api-keys-content">
                  <APIKeyManager />
                </div>
              )}

              {activeTab === 'notifications' && (
                <div id="notifications-content">
                  <NotificationSettings
                    settings={settings.notifications}
                    onUpdate={(updates) => updateSettings('notifications', updates)}
                  />
                </div>
              )}

              {activeTab === 'features' && (
                <div id="feature-flags-content">
                  <FeatureFlagsPanel />
                </div>
              )}

              {activeTab === 'announcements' && (
                <div id="announcements-content">
                  <SystemAnnouncementManager />
                </div>
              )}

              {activeTab === 'audit-logs' && (
                <div id="audit-logs-content">
                  <AuditLogViewer />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </AdminLayout>
  );
};

export default SettingsPage;