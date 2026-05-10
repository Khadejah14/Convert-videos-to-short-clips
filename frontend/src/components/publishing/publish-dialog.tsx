'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ConnectedAccounts } from './connected-accounts';
import {
  PLATFORM_LABELS,
  PLATFORM_COLORS,
  DRAFT_STATUS_LABELS,
  PUBLISH_STATUS_LABELS,
  type Platform,
  type Clip,
  type ConnectedAccount,
  type ExportPreset,
  type PublishDraft,
  type PublishHistoryEntry,
} from '@/types';
import {
  Send,
  Clock,
  Save,
  X,
  ChevronRight,
  Settings2,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Loader2,
} from 'lucide-react';
import toast from 'react-hot-toast';

interface Props {
  clip: Clip;
  jobId: string;
  open: boolean;
  onClose: () => void;
}

type Step = 'platform' | 'details' | 'review' | 'publishing' | 'done';

export function PublishDialog({ clip, jobId, open, onClose }: Props) {
  const { connectedAccounts, exportPresets, setExportPresets } = useAppStore();
  const [step, setStep] = useState<Step>('platform');
  const [selectedPlatform, setSelectedPlatform] = useState<Platform | null>(null);
  const [selectedAccount, setSelectedAccount] = useState<ConnectedAccount | null>(null);
  const [selectedPreset, setSelectedPreset] = useState<ExportPreset | null>(null);

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [visibility, setVisibility] = useState('public');
  const [scheduleDate, setScheduleDate] = useState('');
  const [isScheduling, setIsScheduling] = useState(false);

  const [draft, setDraft] = useState<PublishDraft | null>(null);
  const [historyEntry, setHistoryEntry] = useState<PublishHistoryEntry | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (open) {
      loadPresets();
      setStep('platform');
      setSelectedPlatform(null);
      setSelectedAccount(null);
      setTitle('');
      setDescription('');
      setTags('');
      setScheduleDate('');
      setDraft(null);
      setHistoryEntry(null);
      setPublishing(false);
      setProgress(0);
    }
  }, [open]);

  useEffect(() => {
    if (selectedPlatform) {
      const accounts = connectedAccounts.filter((a) => a.platform === selectedPlatform);
      if (accounts.length > 0) {
        setSelectedAccount(accounts[0]);
      }
    }
  }, [selectedPlatform, connectedAccounts]);

  const loadPresets = async () => {
    try {
      const resp = await api.listExportPresets();
      setExportPresets(resp.presets);
    } catch {}
  };

  const handleSaveDraft = async () => {
    if (!selectedPlatform) return;
    try {
      const draftData = {
        clip_id: clip.id,
        platform: selectedPlatform,
        account_id: selectedAccount?.id,
        preset_id: selectedPreset?.id,
        title,
        description,
        tags: tags || undefined,
        visibility,
      };
      const created = await api.createDraft(draftData);
      setDraft(created);
      toast.success('Draft saved');
    } catch {
      toast.error('Failed to save draft');
    }
  };

  const handlePublishNow = async () => {
    if (!selectedPlatform) return;
    setPublishing(true);
    setStep('publishing');

    try {
      let draftToUse = draft;
      if (!draftToUse) {
        draftToUse = await api.createDraft({
          clip_id: clip.id,
          platform: selectedPlatform,
          account_id: selectedAccount?.id,
          preset_id: selectedPreset?.id,
          title,
          description,
          tags: tags || undefined,
          visibility,
        });
        setDraft(draftToUse);
      }

      const entry = await api.publishNow(draftToUse.id);
      setHistoryEntry(entry);
      setProgress(entry.upload_progress);

      const pollInterval = setInterval(async () => {
        try {
          const updated = await api.getHistoryEntry(entry.id);
          setHistoryEntry(updated);
          setProgress(updated.upload_progress);

          if (updated.status === 'published' || updated.status === 'failed') {
            clearInterval(pollInterval);
            setPublishing(false);
            setStep(updated.status === 'published' ? 'done' : 'review');
            if (updated.status === 'published') {
              toast.success('Published successfully!');
            } else {
              toast.error(`Publishing failed: ${updated.error_message}`);
            }
          }
        } catch {
          clearInterval(pollInterval);
          setPublishing(false);
        }
      }, 2000);
    } catch (err) {
      setPublishing(false);
      setStep('review');
      toast.error(err instanceof Error ? err.message : 'Publishing failed');
    }
  };

  const handleSchedule = async () => {
    if (!draft || !scheduleDate) {
      toast.error('Save draft first and select a date');
      return;
    }
    try {
      await api.createSchedule(draft.id, new Date(scheduleDate).toISOString());
      toast.success('Publishing scheduled');
      onClose();
    } catch {
      toast.error('Failed to schedule');
    }
  };

  const getAvailablePlatforms = (): Platform[] => {
    return ['tiktok', 'youtube', 'instagram'];
  };

  const getPresetsForPlatform = (): ExportPreset[] => {
    if (!selectedPlatform) return [];
    return exportPresets.filter((p) => p.platform === selectedPlatform);
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto mx-4">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <div>
            <CardTitle>Publish Clip</CardTitle>
            <CardDescription>
              Clip #{clip.clip_number} &middot; {Math.round(clip.duration)}s
            </CardDescription>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Step indicator */}
          <div className="flex items-center gap-2 text-sm">
            {(['platform', 'details', 'review'] as Step[]).map((s, i) => (
              <div key={s} className="flex items-center gap-2">
                <div
                  className={`h-6 w-6 rounded-full flex items-center justify-center text-xs font-medium ${
                    step === s
                      ? 'bg-primary text-primary-foreground'
                      : i < ['platform', 'details', 'review'].indexOf(step)
                      ? 'bg-primary/20 text-primary'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {i + 1}
                </div>
                <span className={step === s ? 'font-medium' : 'text-muted-foreground'}>
                  {s === 'platform' ? 'Platform' : s === 'details' ? 'Details' : 'Review'}
                </span>
                {i < 2 && <ChevronRight className="h-4 w-4 text-muted-foreground" />}
              </div>
            ))}
          </div>

          {/* Step: Platform Selection */}
          {step === 'platform' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Select Platform</Label>
                <div className="grid grid-cols-3 gap-3">
                  {getAvailablePlatforms().map((platform) => {
                    const isConnected = connectedAccounts.some(
                      (a) => a.platform === platform
                    );
                    return (
                      <button
                        key={platform}
                        onClick={() => {
                          setSelectedPlatform(platform);
                          if (isConnected) setStep('details');
                        }}
                        className={`p-4 rounded-lg border text-center transition-all ${
                          selectedPlatform === platform
                            ? 'border-primary bg-primary/5'
                            : isConnected
                            ? 'border-border hover:border-primary/50'
                            : 'border-border opacity-60'
                        }`}
                      >
                        <p className="font-medium text-sm">{PLATFORM_LABELS[platform]}</p>
                        {!isConnected && (
                          <p className="text-xs text-muted-foreground mt-1">Not connected</p>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {selectedPlatform &&
                !connectedAccounts.some((a) => a.platform === selectedPlatform) && (
                  <div className="space-y-2">
                    <Label>Connect Account</Label>
                    <ConnectedAccounts
                      selectable
                      onAccountSelect={(acc) => {
                        setSelectedAccount(acc);
                        setStep('details');
                      }}
                    />
                  </div>
                )}

              {selectedPlatform &&
                connectedAccounts.some((a) => a.platform === selectedPlatform) && (
                  <Button onClick={() => setStep('details')} className="w-full">
                    Continue
                    <ChevronRight className="h-4 w-4 ml-2" />
                  </Button>
                )}
            </div>
          )}

          {/* Step: Details */}
          {step === 'details' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter a catchy title..."
                  maxLength={selectedPlatform === 'youtube' ? 100 : 150}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Write a description..."
                  className="w-full min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                  rows={3}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="tags">Tags (comma-separated)</Label>
                <Input
                  id="tags"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="shorts, viral, funny"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Visibility</Label>
                  <select
                    value={visibility}
                    onChange={(e) => setVisibility(e.target.value)}
                    className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
                  >
                    <option value="public">Public</option>
                    <option value="unlisted">Unlisted</option>
                    <option value="private">Private</option>
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>Export Preset</Label>
                  <select
                    value={selectedPreset?.id || ''}
                    onChange={(e) => {
                      const preset = exportPresets.find((p) => p.id === e.target.value);
                      setSelectedPreset(preset || null);
                    }}
                    className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
                  >
                    <option value="">Default</option>
                    {getPresetsForPlatform().map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setStep('platform')} className="flex-1">
                  Back
                </Button>
                <Button variant="secondary" onClick={handleSaveDraft} className="flex-1">
                  <Save className="h-4 w-4 mr-1" />
                  Save Draft
                </Button>
                <Button onClick={() => setStep('review')} className="flex-1">
                  Continue
                  <ChevronRight className="h-4 w-4 ml-2" />
                </Button>
              </div>
            </div>
          )}

          {/* Step: Review */}
          {step === 'review' && (
            <div className="space-y-4">
              <div className="rounded-lg border p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Platform</span>
                  <Badge variant="secondary" className={PLATFORM_COLORS[selectedPlatform!]}>
                    {PLATFORM_LABELS[selectedPlatform!]}
                  </Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Account</span>
                  <span className="text-sm font-medium">
                    {selectedAccount?.display_name || 'Not selected'}
                  </span>
                </div>
                {title && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Title</span>
                    <span className="text-sm font-medium truncate max-w-[250px]">{title}</span>
                  </div>
                )}
                {description && (
                  <div className="flex items-start justify-between">
                    <span className="text-sm text-muted-foreground">Description</span>
                    <span className="text-sm text-right truncate max-w-[250px]">
                      {description.slice(0, 100)}{description.length > 100 ? '...' : ''}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Visibility</span>
                  <span className="text-sm font-medium capitalize">{visibility}</span>
                </div>
              </div>

              {/* Schedule */}
              <div className="rounded-lg border p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <Label>Schedule for later</Label>
                </div>
                <div className="flex gap-2">
                  <Input
                    type="datetime-local"
                    value={scheduleDate}
                    onChange={(e) => setScheduleDate(e.target.value)}
                    min={new Date().toISOString().slice(0, 16)}
                  />
                  <Button
                    variant="outline"
                    onClick={handleSchedule}
                    disabled={!scheduleDate || !draft}
                  >
                    Schedule
                  </Button>
                </div>
              </div>

              <div className="flex gap-2">
                <Button variant="outline" onClick={() => setStep('details')} className="flex-1">
                  Back
                </Button>
                <Button
                  variant="gradient"
                  onClick={handlePublishNow}
                  disabled={publishing || !selectedAccount}
                  className="flex-1"
                >
                  <Send className="h-4 w-4 mr-1" />
                  Publish Now
                </Button>
              </div>
            </div>
          )}

          {/* Step: Publishing Progress */}
          {step === 'publishing' && (
            <div className="space-y-6 py-8 text-center">
              <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
              <div>
                <p className="text-lg font-medium">Publishing to {PLATFORM_LABELS[selectedPlatform!]}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  {historyEntry?.status === 'uploading' ? 'Uploading video...' : 'Processing...'}
                </p>
              </div>
              <div className="max-w-xs mx-auto">
                <Progress value={progress} />
                <p className="text-xs text-muted-foreground mt-2">{Math.round(progress)}% complete</p>
              </div>
            </div>
          )}

          {/* Step: Done */}
          {step === 'done' && historyEntry && (
            <div className="space-y-6 py-8 text-center">
              <div className="h-16 w-16 rounded-full bg-emerald-500/10 flex items-center justify-center mx-auto">
                <CheckCircle2 className="h-8 w-8 text-emerald-400" />
              </div>
              <div>
                <p className="text-lg font-medium">Published Successfully!</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Your clip is now live on {PLATFORM_LABELS[selectedPlatform!]}
                </p>
              </div>
              {historyEntry.platform_post_url && (
                <a
                  href={historyEntry.platform_post_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                >
                  View post <ExternalLink className="h-3 w-3" />
                </a>
              )}
              <Button onClick={onClose} className="w-full">
                Done
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
