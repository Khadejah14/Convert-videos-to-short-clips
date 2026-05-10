'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/layout/sidebar';
import { ConnectedAccounts } from '@/components/publishing/connected-accounts';
import { ExportPresets } from '@/components/publishing/export-presets';
import { PublishingHistory } from '@/components/publishing/publishing-history';
import { ScheduledPublishes } from '@/components/publishing/scheduled-publishes';
import { AnalyticsSummaryCard } from '@/components/publishing/analytics-summary';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Link2,
  Settings2,
  History,
  Clock,
  BarChart3,
  Send,
} from 'lucide-react';

type Tab = 'accounts' | 'presets' | 'history' | 'scheduled' | 'analytics';

export default function PublishPage() {
  const [activeTab, setActiveTab] = useState<Tab>('accounts');

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="lg:ml-64 p-4 lg:p-8">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Publish</h1>
            <p className="text-muted-foreground">
              Connect accounts, manage presets, and publish clips to TikTok, YouTube Shorts, and Instagram Reels
            </p>
          </div>

          {/* Analytics at the top */}
          <div className="mb-8">
            <AnalyticsSummaryCard />
          </div>

          {/* Tabs */}
          <div className="space-y-6">
            <div className="flex gap-1 border-b overflow-x-auto">
              {[
                { id: 'accounts' as Tab, label: 'Connected Accounts', icon: Link2 },
                { id: 'presets' as Tab, label: 'Export Presets', icon: Settings2 },
                { id: 'history' as Tab, label: 'History', icon: History },
                { id: 'scheduled' as Tab, label: 'Scheduled', icon: Clock },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                    activeTab === tab.id
                      ? 'border-primary text-primary'
                      : 'border-transparent text-muted-foreground hover:text-foreground'
                  }`}
                >
                  <tab.icon className="h-4 w-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {activeTab === 'accounts' && <ConnectedAccounts />}
            {activeTab === 'presets' && <ExportPresets />}
            {activeTab === 'history' && <PublishingHistory />}
            {activeTab === 'scheduled' && <ScheduledPublishes />}
          </div>
        </div>
      </main>
    </div>
  );
}
