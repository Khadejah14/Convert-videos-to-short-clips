import type {
  Job,
  JobListResponse,
  UploadConfig,
  ConnectedAccount,
  ConnectedAccountListResponse,
  OAuthInitResponse,
  ExportPreset,
  ExportPresetCreate,
  ExportPresetListResponse,
  PublishDraft,
  PublishDraftCreate,
  PublishDraftListResponse,
  PublishSchedule,
  PublishScheduleListResponse,
  PublishHistoryEntry,
  PublishHistoryListResponse,
  AnalyticsSummary,
  AnalyticsSnapshot,
  Platform,
  ScheduleStatus,
} from '@/types';

const API_BASE = '/api/v1';

class ApiClient {
  private async fetch<T>(url: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async createJob(file: File, config: UploadConfig): Promise<Job> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('clip_count', config.clip_count.toString());
    formData.append('clip_length', config.clip_length.toString());
    formData.append('caption_style', config.caption_style);
    formData.append('use_vision', config.use_vision.toString());

    const response = await fetch(`${API_BASE}/jobs`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async getJob(jobId: string): Promise<Job> {
    return this.fetch<Job>(`/jobs/${jobId}`);
  }

  async listJobs(page = 1, perPage = 20): Promise<JobListResponse> {
    return this.fetch<JobListResponse>(`/jobs?page=${page}&per_page=${perPage}`);
  }

  async cancelJob(jobId: string): Promise<Job> {
    return this.fetch<Job>(`/jobs/${jobId}/cancel`, { method: 'POST' });
  }

  async deleteJob(jobId: string): Promise<void> {
    await fetch(`${API_BASE}/jobs/${jobId}`, { method: 'DELETE' });
  }

  getClipDownloadUrl(jobId: string, clipId: string, type: 'original' | 'final'): string {
    return `${API_BASE}/jobs/${jobId}/clips/${clipId}/${type}`;
  }

  // --- OAuth / Connected Accounts ---

  async getOAuthUrl(platform: Platform): Promise<OAuthInitResponse> {
    return this.fetch<OAuthInitResponse>(`/publishing/oauth/${platform}/url`);
  }

  async listConnectedAccounts(platform?: Platform): Promise<ConnectedAccountListResponse> {
    const params = platform ? `?platform=${platform}` : '';
    return this.fetch<ConnectedAccountListResponse>(`/publishing/accounts${params}`);
  }

  async disconnectAccount(accountId: string): Promise<void> {
    await fetch(`${API_BASE}/publishing/accounts/${accountId}`, { method: 'DELETE' });
  }

  // --- Export Presets ---

  async listExportPresets(platform?: Platform): Promise<ExportPresetListResponse> {
    const params = platform ? `?platform=${platform}` : '';
    return this.fetch<ExportPresetListResponse>(`/publishing/presets${params}`);
  }

  async createExportPreset(data: ExportPresetCreate): Promise<ExportPreset> {
    return this.fetch<ExportPreset>('/publishing/presets', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateExportPreset(
    presetId: string,
    data: Partial<ExportPresetCreate>
  ): Promise<ExportPreset> {
    return this.fetch<ExportPreset>(`/publishing/presets/${presetId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteExportPreset(presetId: string): Promise<void> {
    await fetch(`${API_BASE}/publishing/presets/${presetId}`, { method: 'DELETE' });
  }

  // --- Publish Drafts ---

  async createDraft(data: PublishDraftCreate): Promise<PublishDraft> {
    return this.fetch<PublishDraft>('/publishing/drafts', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async listDrafts(
    platform?: Platform,
    page = 1,
    perPage = 20
  ): Promise<PublishDraftListResponse> {
    const params = new URLSearchParams();
    if (platform) params.set('platform', platform);
    params.set('page', page.toString());
    params.set('per_page', perPage.toString());
    return this.fetch<PublishDraftListResponse>(`/publishing/drafts?${params}`);
  }

  async getDraft(draftId: string): Promise<PublishDraft> {
    return this.fetch<PublishDraft>(`/publishing/drafts/${draftId}`);
  }

  async updateDraft(
    draftId: string,
    data: Partial<PublishDraftCreate & { status: string }>
  ): Promise<PublishDraft> {
    return this.fetch<PublishDraft>(`/publishing/drafts/${draftId}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteDraft(draftId: string): Promise<void> {
    await fetch(`${API_BASE}/publishing/drafts/${draftId}`, { method: 'DELETE' });
  }

  // --- Publish Actions ---

  async publishNow(draftId: string): Promise<PublishHistoryEntry> {
    return this.fetch<PublishHistoryEntry>('/publishing/publish', {
      method: 'POST',
      body: JSON.stringify({ draft_id: draftId }),
    });
  }

  // --- Schedules ---

  async createSchedule(draftId: string, scheduledAt: string): Promise<PublishSchedule> {
    return this.fetch<PublishSchedule>('/publishing/schedules', {
      method: 'POST',
      body: JSON.stringify({ draft_id: draftId, scheduled_at: scheduledAt }),
    });
  }

  async listSchedules(status?: ScheduleStatus): Promise<PublishScheduleListResponse> {
    const params = status ? `?status=${status}` : '';
    return this.fetch<PublishScheduleListResponse>(`/publishing/schedules${params}`);
  }

  async cancelSchedule(scheduleId: string): Promise<void> {
    await fetch(`${API_BASE}/publishing/schedules/${scheduleId}/cancel`, { method: 'POST' });
  }

  // --- Publishing History ---

  async listPublishHistory(
    platform?: Platform,
    page = 1,
    perPage = 20
  ): Promise<PublishHistoryListResponse> {
    const params = new URLSearchParams();
    if (platform) params.set('platform', platform);
    params.set('page', page.toString());
    params.set('per_page', perPage.toString());
    return this.fetch<PublishHistoryListResponse>(`/publishing/history?${params}`);
  }

  async getHistoryEntry(historyId: string): Promise<PublishHistoryEntry> {
    return this.fetch<PublishHistoryEntry>(`/publishing/history/${historyId}`);
  }

  async retryPublish(historyId: string): Promise<PublishHistoryEntry> {
    return this.fetch<PublishHistoryEntry>(`/publishing/history/${historyId}/retry`, {
      method: 'POST',
    });
  }

  // --- Analytics ---

  async getAnalyticsSummary(
    platform?: Platform,
    days = 30
  ): Promise<AnalyticsSummary> {
    const params = new URLSearchParams();
    if (platform) params.set('platform', platform);
    params.set('days', days.toString());
    return this.fetch<AnalyticsSummary>(`/publishing/analytics/summary?${params}`);
  }

  async getPostAnalytics(historyId: string): Promise<AnalyticsSnapshot[]> {
    return this.fetch<AnalyticsSnapshot[]>(`/publishing/analytics/posts/${historyId}`);
  }

  async collectAnalytics(historyId: string): Promise<AnalyticsSnapshot | null> {
    return this.fetch<AnalyticsSnapshot | null>(
      `/publishing/analytics/collect/${historyId}`,
      { method: 'POST' }
    );
  }
}

export const api = new ApiClient();
