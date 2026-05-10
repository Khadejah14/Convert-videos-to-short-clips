'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { useJobPolling } from '@/hooks/useJobs';
import { useAppStore } from '@/lib/store';
import { Sidebar } from '@/components/layout/sidebar';
import { VideoPlayer } from '@/components/player/video-player';
import { ProcessingProgress } from '@/components/ui/processing-progress';
import { PublishDialog } from '@/components/publishing/publish-dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  formatDate,
  formatDuration,
  getStatusColor,
  getStatusBgColor,
  cn,
} from '@/lib/utils';
import { JOB_STATUS_LABELS, CLIP_CATEGORY_COLORS } from '@/types';
import type { Job, Clip } from '@/types';
import {
  ArrowLeft,
  Download,
  Trash2,
  XCircle,
  CheckCircle2,
  Film,
  Clock,
  Scissors,
  Eye,
  Trophy,
  Star,
  Send,
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function JobDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.id as string;

  const { currentJob, setCurrentJob, updateJob, removeJob, isLoading, setLoading } =
    useAppStore();
  const [selectedClip, setSelectedClip] = useState<Clip | null>(null);
  const [showPublishDialog, setShowPublishDialog] = useState(false);

  // Enable polling for active jobs
  const isActive = currentJob
    ? !['completed', 'failed', 'cancelled'].includes(currentJob.status)
    : false;
  useJobPolling(jobId, isActive);

  useEffect(() => {
    const fetchJob = async () => {
      setLoading(true);
      try {
        const job = await api.getJob(jobId);
        setCurrentJob(job);
        if (job.clips.length > 0) {
          setSelectedClip(job.clips[0]);
        }
      } catch (error) {
        toast.error('Failed to load job details');
        router.push('/dashboard');
      } finally {
        setLoading(false);
      }
    };

    fetchJob();
  }, [jobId, setCurrentJob, setLoading, router]);

  const handleCancel = async () => {
    try {
      const job = await api.cancelJob(jobId);
      updateJob(job);
      toast.success('Job cancelled');
    } catch (error) {
      toast.error('Failed to cancel job');
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this job?')) return;

    try {
      await api.deleteJob(jobId);
      removeJob(jobId);
      toast.success('Job deleted');
      router.push('/dashboard');
    } catch (error) {
      toast.error('Failed to delete job');
    }
  };

  const handleDownload = (clip: Clip, type: 'original' | 'final') => {
    const url = api.getClipDownloadUrl(jobId, clip.id, type);
    window.open(url, '_blank');
  };

  if (isLoading || !currentJob) {
    return (
      <div className="min-h-screen bg-background">
        <Sidebar />
        <main className="lg:ml-64 p-4 lg:p-8">
          <div className="max-w-7xl mx-auto">
            <div className="flex items-center justify-center h-[60vh]">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
            </div>
          </div>
        </main>
      </div>
    );
  }

  const isProcessing = !['completed', 'failed', 'cancelled'].includes(
    currentJob.status
  );
  const completedClips = currentJob.clips.filter(
    (c) => c.status === 'completed'
  ).length;

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />

      <main className="lg:ml-64 p-4 lg:p-8">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-6">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push('/dashboard')}
              className="mb-4"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Dashboard
            </Button>

            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold mb-1">
                  {currentJob.original_filename}
                </h1>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusBgColor(currentJob.status)}>
                    {JOB_STATUS_LABELS[currentJob.status]}
                  </Badge>
                  <span className="text-sm text-muted-foreground">
                    {formatDate(currentJob.created_at)}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {isProcessing && (
                  <Button variant="destructive" size="sm" onClick={handleCancel}>
                    <XCircle className="h-4 w-4 mr-2" />
                    Cancel
                  </Button>
                )}
                {!isProcessing && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleDelete}
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Processing Progress */}
          {isProcessing && (
            <Card className="mb-6">
              <CardContent className="p-6">
                <ProcessingProgress
                  status={currentJob.status}
                  progress={currentJob.progress}
                  currentStep={currentJob.current_step}
                  clipsProcessed={completedClips}
                  totalClips={currentJob.clip_count}
                />
              </CardContent>
            </Card>
          )}

          {/* Error Message */}
          {currentJob.status === 'failed' && currentJob.error_message && (
            <Card className="mb-6 border-red-500/50">
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <XCircle className="h-5 w-5 text-red-400 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-400">
                      Processing Failed
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {currentJob.error_message}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Clips */}
          {currentJob.clips.length > 0 && (
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Video Player */}
              <div className="lg:col-span-2">
                <Card>
                  <CardContent className="p-0">
                    {selectedClip?.final_url ? (
                      <VideoPlayer
                        src={selectedClip.final_url}
                        title={selectedClip.category}
                        category={selectedClip.category}
                        duration={selectedClip.duration}
                        onDownload={() =>
                          selectedClip && handleDownload(selectedClip, 'final')
                        }
                      />
                    ) : (
                      <div className="aspect-video bg-muted flex items-center justify-center rounded-xl">
                        <div className="text-center">
                          <Film className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                          <p className="text-muted-foreground">
                            {isProcessing
                              ? 'Clip not ready yet'
                              : 'No clip available'}
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Clip Info */}
                {selectedClip && (
                  <Card className="mt-4">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="font-semibold">
                            {selectedClip.category}
                          </h3>
                          <p className="text-sm text-muted-foreground">
                            {formatDuration(selectedClip.start_time)} -{' '}
                            {formatDuration(selectedClip.end_time)}
                            {' • '}
                            {formatDuration(selectedClip.duration)} duration
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          {selectedClip.rank === 1 && (
                            <Badge className="bg-amber-500/20 text-amber-400 border-amber-500/30">
                              <Trophy className="h-3 w-3 mr-1" />
                              Winner
                            </Badge>
                          )}
                          <Badge
                            className={
                              CLIP_CATEGORY_COLORS[selectedClip.category] ||
                              'bg-muted'
                            }
                          >
                            {selectedClip.category}
                          </Badge>
                        </div>
                      </div>

                      {/* Scores */}
                      {(selectedClip.text_score ||
                        selectedClip.visual_score) && (
                        <div className="grid grid-cols-3 gap-4 p-4 bg-muted rounded-lg">
                          {selectedClip.text_score && (
                            <div>
                              <p className="text-xs text-muted-foreground">
                                Text Score
                              </p>
                              <p className="text-lg font-semibold">
                                {selectedClip.text_score.toFixed(1)}
                              </p>
                            </div>
                          )}
                          {selectedClip.visual_score && (
                            <div>
                              <p className="text-xs text-muted-foreground">
                                Visual Score
                              </p>
                              <p className="text-lg font-semibold">
                                {selectedClip.visual_score.toFixed(1)}
                              </p>
                            </div>
                          )}
                          {selectedClip.combined_score && (
                            <div>
                              <p className="text-xs text-muted-foreground">
                                Combined
                              </p>
                              <p className="text-lg font-semibold text-primary">
                                {selectedClip.combined_score.toFixed(1)}
                              </p>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Visual Hook */}
                      {selectedClip.visual_hook && (
                        <div className="mt-4 p-3 bg-primary/5 rounded-lg border border-primary/20">
                          <div className="flex items-start gap-2">
                            <Eye className="h-4 w-4 text-primary mt-0.5" />
                            <div>
                              <p className="text-xs font-medium text-primary">
                                Visual Hook
                              </p>
                              <p className="text-sm text-muted-foreground mt-1">
                                {selectedClip.visual_hook}
                              </p>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Download Buttons */}
                      <div className="flex gap-2 mt-4">
                        {selectedClip.final_url && (
                          <Button
                            variant="default"
                            size="sm"
                            onClick={() =>
                              handleDownload(selectedClip, 'final')
                            }
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Download Final
                          </Button>
                        )}
                        {selectedClip.original_url && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() =>
                              handleDownload(selectedClip, 'original')
                            }
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Download Original
                          </Button>
                        )}
                        {selectedClip.status === 'completed' && selectedClip.final_url && (
                          <Button
                            variant="gradient"
                            size="sm"
                            onClick={() => setShowPublishDialog(true)}
                          >
                            <Send className="h-4 w-4 mr-2" />
                            Publish
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* Clip List */}
              <div>
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">
                      Generated Clips ({currentJob.clips.length})
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <div className="divide-y">
                      {currentJob.clips.map((clip) => (
                        <button
                          key={clip.id}
                          onClick={() => setSelectedClip(clip)}
                          className={cn(
                            'w-full p-4 text-left transition-colors hover:bg-accent',
                            selectedClip?.id === clip.id && 'bg-accent'
                          )}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                Clip {clip.clip_number}
                              </span>
                              {clip.rank === 1 && (
                                <Star className="h-4 w-4 text-amber-400" />
                              )}
                            </div>
                            <Badge
                              variant="secondary"
                              className={cn(
                                'text-xs',
                                clip.status === 'completed'
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : clip.status === 'failed'
                                  ? 'bg-red-500/20 text-red-400'
                                  : 'bg-muted'
                              )}
                            >
                              {clip.status}
                            </Badge>
                          </div>

                          <p className="text-sm text-muted-foreground mb-2">
                            {clip.category}
                          </p>

                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {formatDuration(clip.duration)}
                            </span>
                            <span className="flex items-center gap-1">
                              <Scissors className="h-3 w-3" />
                              {formatDuration(clip.start_time)} -{' '}
                              {formatDuration(clip.end_time)}
                            </span>
                          </div>

                          {clip.combined_score && (
                            <div className="mt-2">
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="text-muted-foreground">
                                  Score
                                </span>
                                <span className="font-medium">
                                  {clip.combined_score.toFixed(1)}
                                </span>
                              </div>
                              <div className="w-full h-1 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary rounded-full"
                                  style={{
                                    width: `${(clip.combined_score / 10) * 100}%`,
                                  }}
                                />
                              </div>
                            </div>
                          )}
                        </button>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Job Settings */}
                <Card className="mt-4">
                  <CardHeader>
                    <CardTitle className="text-lg">Settings</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Clip Length</span>
                      <span className="font-medium">
                        {currentJob.clip_length}s
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        Caption Style
                      </span>
                      <span className="font-medium capitalize">
                        {currentJob.caption_style}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">
                        Vision Analysis
                      </span>
                      <span className="font-medium">
                        {currentJob.use_vision ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {/* No Clips State */}
          {currentJob.clips.length === 0 && !isProcessing && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16">
                <div className="h-16 w-16 rounded-full bg-muted flex items-center justify-center mb-4">
                  <Scissors className="h-8 w-8 text-muted-foreground" />
                </div>
                <p className="text-lg font-medium mb-1">No clips generated</p>
                <p className="text-sm text-muted-foreground">
                  The processing didn&apos;t produce any valid clips
                </p>
              </CardContent>
            </Card>
          )}

          {/* Publish Dialog */}
          {selectedClip && (
            <PublishDialog
              clip={selectedClip}
              jobId={jobId}
              open={showPublishDialog}
              onClose={() => setShowPublishDialog(false)}
            />
          )}
        </div>
      </main>
    </div>
  );
}
