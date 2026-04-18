"use client";

import { useState } from "react";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Key,
  Bell,
  Download,
  Settings as SettingsIcon,
  Eye,
  EyeOff,
  Save,
  RefreshCw,
  Trash2,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useHealthCheck } from "@/lib/hooks";
import { staggerContainer, staggerItem } from "@/components/ui/motion";

interface SettingsState {
  apiKey: string;
  emailNotifications: boolean;
  highRiskAlerts: boolean;
  analysisComplete: boolean;
  exportFormat: "pdf" | "json" | "csv";
  includeHighlights: boolean;
  includeRecommendations: boolean;
  autoAnalyze: boolean;
  detailedAnalysis: boolean;
}

export default function SettingsPage() {
  const [showApiKey, setShowApiKey] = useState(false);
  const [settings, setSettings] = useState<SettingsState>({
    apiKey: "",
    emailNotifications: true,
    highRiskAlerts: true,
    analysisComplete: false,
    exportFormat: "pdf",
    includeHighlights: true,
    includeRecommendations: true,
    autoAnalyze: false,
    detailedAnalysis: true,
  });

  const { data: health, isLoading: healthLoading } = useHealthCheck();

  const handleSave = () => {
    toast.success("Settings saved successfully");
  };

  const handleReset = () => {
    setSettings({
      apiKey: "",
      emailNotifications: true,
      highRiskAlerts: true,
      analysisComplete: false,
      exportFormat: "pdf",
      includeHighlights: true,
      includeRecommendations: true,
      autoAnalyze: false,
      detailedAnalysis: true,
    });
    toast.success("Settings reset to defaults");
  };

  const ToggleOption = ({
    label,
    description,
    checked,
    onChange,
  }: {
    label: string;
    description: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
  }) => (
    <div className="flex items-start justify-between py-3">
      <div>
        <p className="text-sm font-medium text-gray-900">{label}</p>
        <p className="text-xs text-gray-500">{description}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          checked ? "bg-blue-600" : "bg-gray-200"
        )}
      >
        <span
          className={cn(
            "pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition",
            checked ? "translate-x-4" : "translate-x-0"
          )}
        />
      </button>
    </div>
  );

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900 mb-2">Settings</h1>
        <p className="text-gray-600">
          Manage your API keys, notification preferences, and export options.
        </p>
      </div>

      <motion.div
        className="space-y-6"
        variants={staggerContainer}
        initial="hidden"
        animate="show"
      >
        {/* System Status */}
        <motion.div variants={staggerItem}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <SettingsIcon className="h-4 w-4" />
                System Status
              </CardTitle>
              <CardDescription>Current system health and connectivity.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between py-2">
                <span className="text-sm text-gray-700">API Status</span>
                {healthLoading ? (
                  <Badge variant="secondary">Checking...</Badge>
                ) : health?.status === "healthy" ? (
                  <Badge variant="success">Connected</Badge>
                ) : (
                  <Badge variant="danger">Disconnected</Badge>
                )}
              </div>
              <div className="flex items-center justify-between py-2">
                <span className="text-sm text-gray-700">Version</span>
                <span className="text-sm text-gray-500">{health?.version || "2.0.0"}</span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* API Key Management */}
        <motion.div variants={staggerItem}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Key className="h-4 w-4" />
                API Key
              </CardTitle>
              <CardDescription>Manage your API key for programmatic access.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Input
                    type={showApiKey ? "text" : "password"}
                    value={settings.apiKey}
                    onChange={(e) => setSettings({ ...settings, apiKey: e.target.value })}
                    placeholder="Enter your API key"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
                <Button variant="outline">Generate New</Button>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Keep your API key secure. Do not share it publicly.
              </p>
            </CardContent>
          </Card>
        </motion.div>

        {/* Notification Preferences */}
        <motion.div variants={staggerItem}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Bell className="h-4 w-4" />
                Notifications
              </CardTitle>
              <CardDescription>Configure how you receive notifications.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1">
              <ToggleOption
                label="Email Notifications"
                description="Receive email updates about your analyses"
                checked={settings.emailNotifications}
                onChange={(checked) => setSettings({ ...settings, emailNotifications: checked })}
              />
              <Separator />
              <ToggleOption
                label="High Risk Alerts"
                description="Get notified immediately when high-risk clauses are detected"
                checked={settings.highRiskAlerts}
                onChange={(checked) => setSettings({ ...settings, highRiskAlerts: checked })}
              />
              <Separator />
              <ToggleOption
                label="Analysis Complete"
                description="Receive a notification when analysis finishes"
                checked={settings.analysisComplete}
                onChange={(checked) => setSettings({ ...settings, analysisComplete: checked })}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Export Preferences */}
        <motion.div variants={staggerItem}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Download className="h-4 w-4" />
                Export Options
              </CardTitle>
              <CardDescription>Configure default export settings.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Default Export Format
                </label>
                <Select
                  value={settings.exportFormat}
                  onValueChange={(value: "pdf" | "json" | "csv") =>
                    setSettings({ ...settings, exportFormat: value })
                  }
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pdf">PDF Document</SelectItem>
                    <SelectItem value="json">JSON Data</SelectItem>
                    <SelectItem value="csv">CSV Spreadsheet</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Separator />
              <ToggleOption
                label="Include Clause Highlights"
                description="Highlight detected clauses in exported documents"
                checked={settings.includeHighlights}
                onChange={(checked) => setSettings({ ...settings, includeHighlights: checked })}
              />
              <Separator />
              <ToggleOption
                label="Include Recommendations"
                description="Add AI-generated recommendations to exports"
                checked={settings.includeRecommendations}
                onChange={(checked) => setSettings({ ...settings, includeRecommendations: checked })}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Analysis Defaults */}
        <motion.div variants={staggerItem}>
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <SettingsIcon className="h-4 w-4" />
                Analysis Defaults
              </CardTitle>
              <CardDescription>Configure default analysis behavior.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1">
              <ToggleOption
                label="Auto-analyze on Upload"
                description="Automatically start analysis when documents are uploaded"
                checked={settings.autoAnalyze}
                onChange={(checked) => setSettings({ ...settings, autoAnalyze: checked })}
              />
              <Separator />
              <ToggleOption
                label="Detailed Analysis"
                description="Include comprehensive analysis with recommendations"
                checked={settings.detailedAnalysis}
                onChange={(checked) => setSettings({ ...settings, detailedAnalysis: checked })}
              />
            </CardContent>
          </Card>
        </motion.div>

        {/* Danger Zone */}
        <motion.div variants={staggerItem}>
          <Card className="border-red-200">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2 text-red-700">
                <AlertTriangle className="h-4 w-4" />
                Danger Zone
              </CardTitle>
              <CardDescription>Irreversible actions for your account.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Delete All Data</p>
                  <p className="text-xs text-gray-500">
                    Permanently delete all documents and analyses
                  </p>
                </div>
                <Button variant="destructive" size="sm">
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Action Buttons */}
        <motion.div variants={staggerItem} className="flex items-center justify-between pt-4">
          <Button variant="outline" onClick={handleReset}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button onClick={handleSave}>
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}
