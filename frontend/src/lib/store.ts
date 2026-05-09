import { create } from 'zustand';
import type { Job, UploadConfig } from '@/types';

interface AppState {
  jobs: Job[];
  currentJob: Job | null;
  isLoading: boolean;
  error: string | null;
  
  uploadConfig: UploadConfig;
  setUploadConfig: (config: Partial<UploadConfig>) => void;
  
  setJobs: (jobs: Job[]) => void;
  addJob: (job: Job) => void;
  updateJob: (job: Job) => void;
  setCurrentJob: (job: Job | null) => void;
  removeJob: (jobId: string) => void;
  
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  jobs: [],
  currentJob: null,
  isLoading: false,
  error: null,
  
  uploadConfig: {
    clip_count: 3,
    clip_length: 30,
    caption_style: 'default',
    use_vision: false,
  },
  
  setUploadConfig: (config) =>
    set((state) => ({
      uploadConfig: { ...state.uploadConfig, ...config },
    })),
  
  setJobs: (jobs) => set({ jobs }),
  
  addJob: (job) =>
    set((state) => ({
      jobs: [job, ...state.jobs],
    })),
  
  updateJob: (updatedJob) =>
    set((state) => ({
      jobs: state.jobs.map((job) =>
        job.id === updatedJob.id ? updatedJob : job
      ),
      currentJob:
        state.currentJob?.id === updatedJob.id
          ? updatedJob
          : state.currentJob,
    })),
  
  setCurrentJob: (job) => set({ currentJob: job }),
  
  removeJob: (jobId) =>
    set((state) => ({
      jobs: state.jobs.filter((job) => job.id !== jobId),
      currentJob:
        state.currentJob?.id === jobId ? null : state.currentJob,
    })),
  
  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
}));
