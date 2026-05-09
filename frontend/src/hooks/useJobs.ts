'use client';

import { useEffect, useRef, useCallback } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import type { Job } from '@/types';

export function useJobPolling(jobId: string | null, enabled = true) {
  const updateJob = useAppStore((state) => state.updateJob);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const pollJob = useCallback(async () => {
    if (!jobId) return;

    try {
      const job = await api.getJob(jobId);
      updateJob(job);

      if (['completed', 'failed', 'cancelled'].includes(job.status)) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (error) {
      console.error('Polling error:', error);
    }
  }, [jobId, updateJob]);

  useEffect(() => {
    if (!jobId || !enabled) return;

    pollJob();
    intervalRef.current = setInterval(pollJob, 3000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [jobId, enabled, pollJob]);

  return { pollJob };
}

export function useJobsList() {
  const { jobs, setJobs, setLoading, setError } = useAppStore();

  const fetchJobs = useCallback(async (page = 1) => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.listJobs(page);
      setJobs(response.jobs);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to fetch jobs');
    } finally {
      setLoading(false);
    }
  }, [setJobs, setLoading, setError]);

  return { jobs, fetchJobs };
}
