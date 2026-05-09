import type { Job, JobListResponse, UploadConfig } from '@/types';

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
}

export const api = new ApiClient();
