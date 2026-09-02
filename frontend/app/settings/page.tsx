"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useHealthCheck } from "@/lib/hooks";

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

function TextSwitch({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-6 border-t border-rule py-5 first:border-t-0">
      <div className="min-w-0">
        <p className="text-sm font-medium text-ink">{label}</p>
        <p className="mt-1 text-sm text-ink-muted">{description}</p>
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="shrink-0 text-sm underline decoration-rule underline-offset-4 hover:text-brass hover:decoration-brass"
      >
        {checked ? "On" : "Off"}
      </button>
    </div>
  );
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

  const apiStatus = healthLoading
    ? "Checking"
    : health?.status === "healthy"
    ? "Connected"
    : "Disconnected";

  return (
    <div className="page-wrap py-12 lg:py-16">
      <div className="mb-10">
        <p className="eyebrow">Workspace</p>
        <h1 className="display mt-3 text-3xl sm:text-4xl">Settings</h1>
        <p className="mt-3 max-w-xl text-base leading-relaxed text-ink-muted">
          Manage API keys, notification preferences, and export options.
        </p>
      </div>

      <section className="border-y border-rule py-8">
        <p className="eyebrow">System</p>
        <dl className="mt-5 grid gap-6 sm:grid-cols-2">
          <div>
            <dt className="text-sm text-ink-muted">API status</dt>
            <dd className="mt-1 font-serif text-xl text-ink">{apiStatus}</dd>
          </div>
          <div>
            <dt className="text-sm text-ink-muted">Version</dt>
            <dd className="mt-1 font-serif text-xl text-ink">{health?.version || "2.0.0"}</dd>
          </div>
        </dl>
        <div className="mt-8">
          <label className="block text-sm font-medium text-ink" htmlFor="api-key">
            API key
          </label>
          <div className="mt-2 flex flex-col gap-3 sm:flex-row">
            <Input
              id="api-key"
              type={showApiKey ? "text" : "password"}
              value={settings.apiKey}
              onChange={(e) => setSettings({ ...settings, apiKey: e.target.value })}
              placeholder="Enter your API key"
              className="flex-1"
            />
            <div className="flex items-center gap-6">
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="text-sm underline decoration-rule underline-offset-4 hover:text-brass hover:decoration-brass"
              >
                {showApiKey ? "Hide" : "Show"}
              </button>
              <button
                type="button"
                className="text-sm underline decoration-rule underline-offset-4 hover:text-brass hover:decoration-brass"
              >
                Generate new
              </button>
            </div>
          </div>
          <p className="mt-2 text-xs text-ink-muted">
            Keep your API key secure. Do not share it publicly.
          </p>
        </div>
      </section>

      <section className="py-8">
        <p className="eyebrow">Notifications</p>
        <div className="mt-4">
          <TextSwitch
            label="Email notifications"
            description="Receive email updates about your analyses"
            checked={settings.emailNotifications}
            onChange={(checked) => setSettings({ ...settings, emailNotifications: checked })}
          />
          <TextSwitch
            label="High risk alerts"
            description="Get notified when high-risk clauses are detected"
            checked={settings.highRiskAlerts}
            onChange={(checked) => setSettings({ ...settings, highRiskAlerts: checked })}
          />
          <TextSwitch
            label="Analysis complete"
            description="Receive a notification when analysis finishes"
            checked={settings.analysisComplete}
            onChange={(checked) => setSettings({ ...settings, analysisComplete: checked })}
          />
        </div>
      </section>

      <section className="border-t border-rule py-8">
        <p className="eyebrow">Export</p>
        <div className="mt-5">
          <label className="block text-sm font-medium text-ink" htmlFor="export-format">
            Default export format
          </label>
          <Select
            value={settings.exportFormat}
            onValueChange={(value: "pdf" | "json" | "csv") =>
              setSettings({ ...settings, exportFormat: value })
            }
          >
            <SelectTrigger id="export-format" className="mt-2 w-full sm:w-[220px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="pdf">PDF document</SelectItem>
              <SelectItem value="json">JSON data</SelectItem>
              <SelectItem value="csv">CSV spreadsheet</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="mt-6">
          <TextSwitch
            label="Include clause highlights"
            description="Highlight detected clauses in exported documents"
            checked={settings.includeHighlights}
            onChange={(checked) => setSettings({ ...settings, includeHighlights: checked })}
          />
          <TextSwitch
            label="Include recommendations"
            description="Add generated recommendations to exports"
            checked={settings.includeRecommendations}
            onChange={(checked) => setSettings({ ...settings, includeRecommendations: checked })}
          />
        </div>
      </section>

      <section className="border-t border-rule py-8">
        <p className="eyebrow">Analysis</p>
        <div className="mt-4">
          <TextSwitch
            label="Auto-analyze on upload"
            description="Start analysis when documents are uploaded"
            checked={settings.autoAnalyze}
            onChange={(checked) => setSettings({ ...settings, autoAnalyze: checked })}
          />
          <TextSwitch
            label="Detailed analysis"
            description="Include comprehensive analysis with recommendations"
            checked={settings.detailedAnalysis}
            onChange={(checked) => setSettings({ ...settings, detailedAnalysis: checked })}
          />
        </div>
      </section>

      <section className="border-y border-rule py-8">
        <p className="eyebrow">Danger zone</p>
        <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-baseline sm:justify-between">
          <div>
            <p className="font-serif text-xl font-medium text-ink">Delete all data</p>
            <p className="mt-1 text-sm text-ink-muted">
              Permanently delete all documents and analyses.
            </p>
          </div>
          <button
            type="button"
            className="text-sm text-red-900 underline decoration-red-900/40 underline-offset-4 hover:decoration-red-900"
          >
            Delete all data
          </button>
        </div>
      </section>

      <div className="flex flex-col-reverse items-start gap-4 pt-8 sm:flex-row sm:items-center sm:justify-between">
        <Button variant="link" onClick={handleReset}>
          Reset to defaults
        </Button>
        <Button onClick={handleSave}>Save changes</Button>
      </div>
    </div>
  );
}
