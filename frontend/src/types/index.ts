export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  current_step: string;
  original_filename: string;
  clip_count: number;
  clip_length: number;
  caption_style: string;
  use_vision: boolean;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  error_message?: string;
  retry_count: number;
  clips: Clip[];
}

export type JobStatus =
  | 'pending'
  | 'extracting_audio'
  | 'transcribing'
  | 'analyzing'
  | 'extracting_clips'
  | 'processing_clips'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface Clip {
  id: string;
  clip_number: number;
  status: ClipStatus;
  category: string;
  start_time: number;
  end_time: number;
  duration: number;
  original_url?: string;
  final_url?: string;
  text_score?: number;
  visual_score?: number;
  combined_score?: number;
  visual_hook?: string;
  rank?: number;
  no_captions: boolean;
  error_message?: string;
}

export type ClipStatus =
  | 'pending'
  | 'extracting'
  | 'cropping'
  | 'captioning'
  | 'completed'
  | 'failed';

export interface JobListResponse {
  jobs: Job[];
  total: number;
  page: number;
  per_page: number;
}

export interface UploadConfig {
  clip_count: number;
  clip_length: number;
  caption_style: string;
  use_vision: boolean;
}

export const JOB_STATUS_LABELS: Record<JobStatus, string> = {
  pending: 'Pending',
  extracting_audio: 'Extracting Audio',
  transcribing: 'Transcribing',
  analyzing: 'Analyzing Content',
  extracting_clips: 'Extracting Clips',
  processing_clips: 'Processing Clips',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

export const JOB_STATUS_PROGRESS: Record<JobStatus, number> = {
  pending: 0,
  extracting_audio: 10,
  transcribing: 25,
  analyzing: 40,
  extracting_clips: 55,
  processing_clips: 60,
  completed: 100,
  failed: 0,
  cancelled: 0,
};

export const CLIP_CATEGORY_COLORS: Record<string, string> = {
  'Hook Focus': 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  'Emotional Peak': 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  'Viral Moment': 'bg-violet-500/20 text-violet-400 border-violet-500/30',
};

export const CAPTION_STYLES = [
  { value: 'default', label: 'Default', description: 'Solid black background' },
  { value: 'minimal', label: 'Minimal', description: 'Transparent, subtle text' },
  { value: 'highlight', label: 'Highlight', description: 'Bold gold emphasis' },
] as const;

// --- Publishing Types ---

export type Platform = 'tiktok' | 'youtube' | 'instagram';

export type AccountStatus = 'active' | 'expired' | 'revoked' | 'error';

export type DraftStatus = 'draft' | 'ready' | 'publishing' | 'published' | 'failed';

export type ScheduleStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';

export type PublishStatus = 'uploading' | 'processing' | 'published' | 'failed';

export interface ConnectedAccount {
  id: string;
  platform: Platform;
  platform_user_id: string;
  platform_username?: string;
  display_name?: string;
  avatar_url?: string;
  status: AccountStatus;
  scopes?: string;
  created_at: string;
  updated_at: string;
  last_used_at?: string;
}

export interface ConnectedAccountListResponse {
  accounts: ConnectedAccount[];
  total: number;
}

export interface OAuthInitResponse {
  authorization_url: string;
  state: string;
}

export interface ExportPreset {
  id: string;
  user_id?: string;
  name: string;
  platform: Platform;
  is_default: boolean;
  is_system: boolean;
  resolution: string;
  fps: number;
  bitrate: string;
  format: string;
  codec: string;
  max_duration: number;
  caption_style: string;
  watermark_enabled: boolean;
  watermark_position: string;
  custom_settings?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ExportPresetCreate {
  name: string;
  platform: Platform;
  resolution?: string;
  fps?: number;
  bitrate?: string;
  format?: string;
  codec?: string;
  max_duration?: number;
  caption_style?: string;
  watermark_enabled?: boolean;
  watermark_position?: string;
  custom_settings?: Record<string, unknown>;
}

export interface ExportPresetListResponse {
  presets: ExportPreset[];
  total: number;
}

export interface PublishDraft {
  id: string;
  user_id: string;
  clip_id: string;
  account_id?: string;
  preset_id?: string;
  platform: Platform;
  status: DraftStatus;
  title: string;
  description: string;
  tags?: string;
  thumbnail_path?: string;
  cover_frame_time?: number;
  visibility: string;
  category?: string;
  language: string;
  publish_config?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PublishDraftCreate {
  clip_id: string;
  platform: Platform;
  account_id?: string;
  preset_id?: string;
  title?: string;
  description?: string;
  tags?: string;
  visibility?: string;
  category?: string;
  language?: string;
  cover_frame_time?: number;
  publish_config?: Record<string, unknown>;
}

export interface PublishDraftListResponse {
  drafts: PublishDraft[];
  total: number;
  page: number;
  per_page: number;
}

export interface PublishSchedule {
  id: string;
  draft_id: string;
  user_id: string;
  scheduled_at: string;
  status: ScheduleStatus;
  retry_count: number;
  error_message?: string;
  created_at: string;
  executed_at?: string;
}

export interface PublishScheduleListResponse {
  schedules: PublishSchedule[];
  total: number;
}

export interface PublishHistoryEntry {
  id: string;
  draft_id: string;
  user_id: string;
  account_id?: string;
  platform: Platform;
  status: PublishStatus;
  platform_post_id?: string;
  platform_post_url?: string;
  title: string;
  description: string;
  tags?: string;
  visibility: string;
  upload_progress: number;
  error_message?: string;
  retry_count: number;
  published_at?: string;
  created_at: string;
  updated_at: string;
}

export interface PublishHistoryListResponse {
  history: PublishHistoryEntry[];
  total: number;
  page: number;
  per_page: number;
}

export interface AnalyticsSnapshot {
  id: string;
  publish_history_id: string;
  platform: Platform;
  views: number;
  likes: number;
  comments: number;
  shares: number;
  watch_time_seconds: number;
  average_watch_time: number;
  engagement_rate: number;
  follower_count: number;
  raw_metrics?: Record<string, unknown>;
  snapshot_at: string;
}

export interface AnalyticsSummary {
  total_views: number;
  total_likes: number;
  total_comments: number;
  total_shares: number;
  total_watch_time: number;
  average_engagement_rate: number;
  posts_count: number;
  platform_breakdown: Record<string, number>;
}

export const PLATFORM_LABELS: Record<Platform, string> = {
  tiktok: 'TikTok',
  youtube: 'YouTube Shorts',
  instagram: 'Instagram Reels',
};

export const PLATFORM_COLORS: Record<Platform, string> = {
  tiktok: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  youtube: 'bg-red-500/20 text-red-400 border-red-500/30',
  instagram: 'bg-violet-500/20 text-violet-400 border-violet-500/30',
};

export const DRAFT_STATUS_LABELS: Record<DraftStatus, string> = {
  draft: 'Draft',
  ready: 'Ready',
  publishing: 'Publishing',
  published: 'Published',
  failed: 'Failed',
};

export const PUBLISH_STATUS_LABELS: Record<PublishStatus, string> = {
  uploading: 'Uploading',
  processing: 'Processing',
  published: 'Published',
  failed: 'Failed',
};
