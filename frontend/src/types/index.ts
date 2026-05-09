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
