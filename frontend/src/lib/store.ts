import { create } from 'zustand';
import type {
  Job,
  UploadConfig,
  ConnectedAccount,
  PublishDraft,
  PublishHistoryEntry,
  ExportPreset,
  PublishSchedule,
  AnalyticsSummary,
} from '@/types';

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

  // Publishing state
  connectedAccounts: ConnectedAccount[];
  setConnectedAccounts: (accounts: ConnectedAccount[]) => void;
  addConnectedAccount: (account: ConnectedAccount) => void;
  removeConnectedAccount: (accountId: string) => void;

  drafts: PublishDraft[];
  setDrafts: (drafts: PublishDraft[]) => void;
  addDraft: (draft: PublishDraft) => void;
  updateDraft: (draft: PublishDraft) => void;
  removeDraft: (draftId: string) => void;

  publishHistory: PublishHistoryEntry[];
  setPublishHistory: (history: PublishHistoryEntry[]) => void;
  addHistoryEntry: (entry: PublishHistoryEntry) => void;
  updateHistoryEntry: (entry: PublishHistoryEntry) => void;

  schedules: PublishSchedule[];
  setSchedules: (schedules: PublishSchedule[]) => void;
  removeSchedule: (scheduleId: string) => void;

  exportPresets: ExportPreset[];
  setExportPresets: (presets: ExportPreset[]) => void;

  analyticsSummary: AnalyticsSummary | null;
  setAnalyticsSummary: (summary: AnalyticsSummary | null) => void;
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

  // Publishing
  connectedAccounts: [],
  setConnectedAccounts: (accounts) => set({ connectedAccounts: accounts }),
  addConnectedAccount: (account) =>
    set((state) => ({
      connectedAccounts: [account, ...state.connectedAccounts],
    })),
  removeConnectedAccount: (accountId) =>
    set((state) => ({
      connectedAccounts: state.connectedAccounts.filter((a) => a.id !== accountId),
    })),

  drafts: [],
  setDrafts: (drafts) => set({ drafts }),
  addDraft: (draft) =>
    set((state) => ({
      drafts: [draft, ...state.drafts],
    })),
  updateDraft: (updatedDraft) =>
    set((state) => ({
      drafts: state.drafts.map((d) =>
        d.id === updatedDraft.id ? updatedDraft : d
      ),
    })),
  removeDraft: (draftId) =>
    set((state) => ({
      drafts: state.drafts.filter((d) => d.id !== draftId),
    })),

  publishHistory: [],
  setPublishHistory: (history) => set({ publishHistory: history }),
  addHistoryEntry: (entry) =>
    set((state) => ({
      publishHistory: [entry, ...state.publishHistory],
    })),
  updateHistoryEntry: (updatedEntry) =>
    set((state) => ({
      publishHistory: state.publishHistory.map((e) =>
        e.id === updatedEntry.id ? updatedEntry : e
      ),
    })),

  schedules: [],
  setSchedules: (schedules) => set({ schedules }),
  removeSchedule: (scheduleId) =>
    set((state) => ({
      schedules: state.schedules.filter((s) => s.id !== scheduleId),
    })),

  exportPresets: [],
  setExportPresets: (presets) => set({ exportPresets: presets }),

  analyticsSummary: null,
  setAnalyticsSummary: (summary) => set({ analyticsSummary: summary }),
}));
