'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useJobsList } from '@/hooks/useJobs';
import { api } from '@/lib/api';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import {
  formatDate,
  getStatusColor,
  getStatusBgColor,
  cn,
} from '@/lib/utils';
import { JOB_STATUS_LABELS } from '@/types';
import type { JobStatus } from '@/types';
import {
  Film,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Search,
  Filter,
  Trash2,
  Loader2,
} from 'lucide-react';
import toast from 'react-hot-toast';

const STATUS_FILTERS: { value: JobStatus | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'completed', label: 'Completed' },
  { value: 'processing_clips', label: 'Processing' },
  { value: 'failed', label: 'Failed' },
];

export default function HistoryPage() {
  const { jobs, fetchJobs } = useJobsList();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<JobStatus | 'all'>('all');
  const [isDeleting, setIsDeleting] = useState<string | null>(null);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const handleDelete = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this job?')) return;

    setIsDeleting(jobId);
    try {
      await api.deleteJob(jobId);
      fetchJobs();
      toast.success('Job deleted');
    } catch (error) {
      toast.error('Failed to delete job');
    } finally {
      setIsDeleting(null);
    }
  };

  const filteredJobs = jobs.filter((job) => {
    const matchesSearch = job.original_filename
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesStatus =
      statusFilter === 'all' || job.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="lg:ml-64 p-4 lg:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold mb-2">Job History</h1>
            <p className="text-muted-foreground">
              View and manage all your video processing jobs
            </p>
          </div>

          {/* Filters */}
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by filename..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="flex gap-2 overflow-x-auto pb-2">
              {STATUS_FILTERS.map((filter) => (
                <Button
                  key={filter.value}
                  variant={
                    statusFilter === filter.value ? 'default' : 'outline'
                  }
                  size="sm"
                  onClick={() => setStatusFilter(filter.value)}
                  className="whitespace-nowrap"
                >
                  {filter.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Jobs List */}
          {filteredJobs.length === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Film className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="text-lg font-medium mb-1">
                  {searchQuery || statusFilter !== 'all'
                    ? 'No matching jobs'
                    : 'No jobs yet'}
                </p>
                <p className="text-sm text-muted-foreground">
                  {searchQuery || statusFilter !== 'all'
                    ? 'Try adjusting your filters'
                    : 'Upload your first video to get started'}
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {filteredJobs.map((job) => {
                const isProcessing = ![
                  'completed',
                  'failed',
                  'cancelled',
                ].includes(job.status);

                return (
                  <Card
                    key={job.id}
                    className="hover:border-primary/50 transition-colors"
                  >
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <Link
                          href={`/dashboard/jobs/${job.id}`}
                          className="flex items-center gap-4 flex-1 min-w-0"
                        >
                          <div
                            className={cn(
                              'h-10 w-10 rounded-lg flex items-center justify-center shrink-0',
                              job.status === 'completed'
                                ? 'bg-emerald-500/10'
                                : job.status === 'failed'
                                ? 'bg-red-500/10'
                                : 'bg-primary/10'
                            )}
                          >
                            {job.status === 'completed' ? (
                              <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                            ) : job.status === 'failed' ? (
                              <XCircle className="h-5 w-5 text-red-400" />
                            ) : (
                              <Loader2 className="h-5 w-5 text-primary animate-spin" />
                            )}
                          </div>

                          <div className="flex-1 min-w-0">
                            <p className="font-medium truncate">
                              {job.original_filename}
                            </p>
                            <div className="flex flex-wrap items-center gap-2 mt-1">
                              <Badge
                                variant="secondary"
                                className={getStatusBgColor(job.status)}
                              >
                                {JOB_STATUS_LABELS[job.status]}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {formatDate(job.created_at)}
                              </span>
                              <span className="text-xs text-muted-foreground">
                                {job.clip_count} clips • {job.clip_length}s each
                              </span>
                              {job.use_vision && (
                                <Badge variant="outline" className="text-xs">
                                  Vision
                                </Badge>
                              )}
                            </div>
                          </div>
                        </Link>

                        <div className="flex items-center gap-2 ml-4">
                          {!isProcessing && (
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleDelete(job.id)}
                              disabled={isDeleting === job.id}
                            >
                              {isDeleting === job.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Trash2 className="h-4 w-4 text-muted-foreground hover:text-red-400" />
                              )}
                            </Button>
                          )}
                          <Link href={`/dashboard/jobs/${job.id}`}>
                            <Button variant="ghost" size="icon">
                              <ArrowRight className="h-4 w-4" />
                            </Button>
                          </Link>
                        </div>
                      </div>

                      {isProcessing && (
                        <div className="mt-3 ml-14">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-muted-foreground">
                              {job.current_step}
                            </span>
                            <span className="font-medium">
                              {Math.round(job.progress)}%
                            </span>
                          </div>
                          <Progress value={job.progress} size="sm" />
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
