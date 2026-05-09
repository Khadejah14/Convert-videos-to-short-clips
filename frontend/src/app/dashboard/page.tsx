'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useJobsList } from '@/hooks/useJobs';
import { useAppStore } from '@/lib/store';
import { api } from '@/lib/api';
import { VideoUpload } from '@/components/upload/video-upload';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  formatDate,
  getStatusColor,
  getStatusBgColor,
  cn,
} from '@/lib/utils';
import { JOB_STATUS_LABELS } from '@/types';
import type { UploadConfig } from '@/types';
import {
  Film,
  Clock,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Sparkles,
  TrendingUp,
  Zap,
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function DashboardPage() {
  const { jobs, fetchJobs } = useJobsList();
  const { addJob, isLoading, setIsLoading, setError } = useAppStore();
  const [showUpload, setShowUpload] = useState(true);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const handleUpload = async (file: File, config: UploadConfig) => {
    setIsLoading(true);
    setError(null);

    try {
      const job = await api.createJob(file, config);
      addJob(job);
      setShowUpload(false);
      toast.success('Video uploaded successfully! Processing started.');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed';
      setError(message);
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  const stats = {
    total: jobs.length,
    completed: jobs.filter((j) => j.status === 'completed').length,
    processing: jobs.filter((j) =>
      !['completed', 'failed', 'cancelled'].includes(j.status)
    ).length,
    failed: jobs.filter((j) => j.status === 'failed').length,
  };

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="lg:ml-64 p-4 lg:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
            <p className="text-muted-foreground">
              Transform your long videos into engaging short clips with AI
            </p>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Jobs</p>
                    <p className="text-2xl font-bold">{stats.total}</p>
                  </div>
                  <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Film className="h-5 w-5 text-primary" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Completed</p>
                    <p className="text-2xl font-bold text-emerald-400">
                      {stats.completed}
                    </p>
                  </div>
                  <div className="h-10 w-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
                    <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Processing</p>
                    <p className="text-2xl font-bold text-blue-400">
                      {stats.processing}
                    </p>
                  </div>
                  <div className="h-10 w-10 rounded-lg bg-blue-500/10 flex items-center justify-center">
                    <Zap className="h-5 w-5 text-blue-400 animate-pulse" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Failed</p>
                    <p className="text-2xl font-bold text-red-400">
                      {stats.failed}
                    </p>
                  </div>
                  <div className="h-10 w-10 rounded-lg bg-red-500/10 flex items-center justify-center">
                    <XCircle className="h-5 w-5 text-red-400" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Upload Section */}
          {showUpload ? (
            <div className="mb-8">
              <VideoUpload onUpload={handleUpload} isUploading={isLoading} />
            </div>
          ) : (
            <div className="mb-8">
              <Button onClick={() => setShowUpload(true)}>
                <Sparkles className="h-4 w-4 mr-2" />
                Upload New Video
              </Button>
            </div>
          )}

          {/* Recent Jobs */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Recent Jobs</h2>
              <Link href="/dashboard/history">
                <Button variant="ghost" size="sm">
                  View all
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </Link>
            </div>

            {jobs.length === 0 ? (
              <Card>
                <CardContent className="flex flex-col items-center justify-center py-16">
                  <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center mb-4">
                    <Film className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <p className="text-lg font-medium mb-1">No jobs yet</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Upload your first video to get started
                  </p>
                  <Button onClick={() => setShowUpload(true)}>
                    <Sparkles className="h-4 w-4 mr-2" />
                    Upload Video
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {jobs.slice(0, 5).map((job) => (
                  <Link key={job.id} href={`/dashboard/jobs/${job.id}`}>
                    <Card className="hover:border-primary/50 transition-colors cursor-pointer">
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4 flex-1 min-w-0">
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
                                <Film className="h-5 w-5 text-primary" />
                              )}
                            </div>

                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate">
                                {job.original_filename}
                              </p>
                              <div className="flex items-center gap-2 mt-1">
                                <Badge
                                  variant="secondary"
                                  className={getStatusBgColor(job.status)}
                                >
                                  {JOB_STATUS_LABELS[job.status]}
                                </Badge>
                                <span className="text-xs text-muted-foreground">
                                  {formatDate(job.created_at)}
                                </span>
                              </div>
                            </div>
                          </div>

                          <div className="hidden sm:flex items-center gap-4 ml-4">
                            {!['completed', 'failed', 'cancelled'].includes(
                              job.status
                            ) && (
                              <div className="w-32">
                                <Progress value={job.progress} size="sm" />
                              </div>
                            )}
                            <ArrowRight className="h-4 w-4 text-muted-foreground" />
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
