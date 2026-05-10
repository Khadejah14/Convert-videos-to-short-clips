'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  PLATFORM_LABELS,
  PLATFORM_COLORS,
  type Platform,
  type ExportPreset,
  type ExportPresetCreate,
} from '@/types';
import {
  Settings2,
  Plus,
  Trash2,
  Edit3,
  Save,
  X,
  Monitor,
} from 'lucide-react';
import toast from 'react-hot-toast';

export function ExportPresets() {
  const { exportPresets, setExportPresets } = useAppStore();
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState<ExportPresetCreate>({
    name: '',
    platform: 'tiktok',
    resolution: '1080x1920',
    fps: 30,
    bitrate: '8M',
    format: 'mp4',
    codec: 'h264',
    max_duration: 60,
    caption_style: 'default',
    watermark_enabled: false,
    watermark_position: 'bottom-right',
  });

  useEffect(() => {
    loadPresets();
  }, []);

  const loadPresets = async () => {
    try {
      const resp = await api.listExportPresets();
      setExportPresets(resp.presets);
    } catch {
      toast.error('Failed to load presets');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!form.name.trim()) {
      toast.error('Preset name is required');
      return;
    }
    try {
      const preset = await api.createExportPreset(form);
      setExportPresets([preset, ...exportPresets]);
      setCreating(false);
      resetForm();
      toast.success('Preset created');
    } catch {
      toast.error('Failed to create preset');
    }
  };

  const handleDelete = async (presetId: string) => {
    try {
      await api.deleteExportPreset(presetId);
      setExportPresets(exportPresets.filter((p) => p.id !== presetId));
      toast.success('Preset deleted');
    } catch {
      toast.error('Failed to delete preset');
    }
  };

  const resetForm = () => {
    setForm({
      name: '',
      platform: 'tiktok',
      resolution: '1080x1920',
      fps: 30,
      bitrate: '8M',
      format: 'mp4',
      codec: 'h264',
      max_duration: 60,
      caption_style: 'default',
      watermark_enabled: false,
      watermark_position: 'bottom-right',
    });
  };

  const startEdit = (preset: ExportPreset) => {
    setEditing(preset.id);
    setForm({
      name: preset.name,
      platform: preset.platform,
      resolution: preset.resolution,
      fps: preset.fps,
      bitrate: preset.bitrate,
      format: preset.format,
      codec: preset.codec,
      max_duration: preset.max_duration,
      caption_style: preset.caption_style,
      watermark_enabled: preset.watermark_enabled,
      watermark_position: preset.watermark_position,
    });
  };

  const handleUpdate = async () => {
    if (!editing) return;
    try {
      const updated = await api.updateExportPreset(editing, form);
      setExportPresets(exportPresets.map((p) => (p.id === editing ? updated : p)));
      setEditing(null);
      resetForm();
      toast.success('Preset updated');
    } catch {
      toast.error('Failed to update preset');
    }
  };

  const PresetForm = ({
    onSubmit,
    onCancel,
  }: {
    onSubmit: () => void;
    onCancel: () => void;
  }) => (
    <div className="space-y-4 border rounded-lg p-4">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Name</Label>
          <Input
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            placeholder="My Custom Preset"
          />
        </div>
        <div className="space-y-2">
          <Label>Platform</Label>
          <select
            value={form.platform}
            onChange={(e) => setForm({ ...form, platform: e.target.value as Platform })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="tiktok">TikTok</option>
            <option value="youtube">YouTube Shorts</option>
            <option value="instagram">Instagram Reels</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label>Resolution</Label>
          <select
            value={form.resolution}
            onChange={(e) => setForm({ ...form, resolution: e.target.value })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="1080x1920">1080x1920 (9:16)</option>
            <option value="1080x1080">1080x1080 (1:1)</option>
            <option value="720x1280">720x1280 (9:16)</option>
          </select>
        </div>
        <div className="space-y-2">
          <Label>FPS</Label>
          <select
            value={form.fps}
            onChange={(e) => setForm({ ...form, fps: parseInt(e.target.value) })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value={24}>24</option>
            <option value={30}>30</option>
            <option value={60}>60</option>
          </select>
        </div>
        <div className="space-y-2">
          <Label>Max Duration (s)</Label>
          <Input
            type="number"
            value={form.max_duration}
            onChange={(e) => setForm({ ...form, max_duration: parseInt(e.target.value) || 60 })}
            min={5}
            max={600}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label>Bitrate</Label>
          <select
            value={form.bitrate}
            onChange={(e) => setForm({ ...form, bitrate: e.target.value })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="4M">4 Mbps</option>
            <option value="8M">8 Mbps</option>
            <option value="12M">12 Mbps</option>
            <option value="16M">16 Mbps</option>
          </select>
        </div>
        <div className="space-y-2">
          <Label>Caption Style</Label>
          <select
            value={form.caption_style}
            onChange={(e) => setForm({ ...form, caption_style: e.target.value })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="default">Default</option>
            <option value="minimal">Minimal</option>
            <option value="highlight">Highlight</option>
          </select>
        </div>
        <div className="space-y-2">
          <Label>Watermark</Label>
          <select
            value={form.watermark_enabled ? 'on' : 'off'}
            onChange={(e) => setForm({ ...form, watermark_enabled: e.target.value === 'on' })}
            className="w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="off">Off</option>
            <option value="on">On</option>
          </select>
        </div>
      </div>

      <div className="flex gap-2 justify-end">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          <X className="h-4 w-4 mr-1" />
          Cancel
        </Button>
        <Button size="sm" onClick={onSubmit}>
          <Save className="h-4 w-4 mr-1" />
          {editing ? 'Update' : 'Create'}
        </Button>
      </div>
    </div>
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Export Presets
            </CardTitle>
            <CardDescription>
              Configure platform-specific export settings
            </CardDescription>
          </div>
          {!creating && !editing && (
            <Button size="sm" onClick={() => setCreating(true)}>
              <Plus className="h-4 w-4 mr-1" />
              New Preset
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {creating && (
          <PresetForm
            onSubmit={handleCreate}
            onCancel={() => {
              setCreating(false);
              resetForm();
            }}
          />
        )}

        {editing && (
          <PresetForm
            onSubmit={handleUpdate}
            onCancel={() => {
              setEditing(null);
              resetForm();
            }}
          />
        )}

        {loading ? (
          <div className="space-y-3">
            {[1, 2].map((i) => (
              <div key={i} className="h-16 bg-muted rounded-lg animate-pulse" />
            ))}
          </div>
        ) : exportPresets.length === 0 && !creating ? (
          <div className="text-center py-8 text-muted-foreground">
            <Monitor className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No custom presets yet</p>
            <p className="text-xs">Create one to customize export settings</p>
          </div>
        ) : (
          <div className="space-y-2">
            {exportPresets.map((preset) => (
              <div
                key={preset.id}
                className="flex items-center justify-between p-3 rounded-lg border"
              >
                <div className="flex items-center gap-3">
                  <Badge variant="secondary" className={PLATFORM_COLORS[preset.platform]}>
                    {PLATFORM_LABELS[preset.platform]}
                  </Badge>
                  <div>
                    <p className="text-sm font-medium">{preset.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {preset.resolution} &middot; {preset.fps}fps &middot; {preset.bitrate}
                      {preset.is_system && ' &middot; System'}
                    </p>
                  </div>
                </div>

                {!preset.is_system && (
                  <div className="flex items-center gap-1">
                    <Button variant="ghost" size="icon" onClick={() => startEdit(preset)}>
                      <Edit3 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDelete(preset.id)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
