'use client';

import { cn } from '@/lib/utils';
import { JOB_STATUS_LABELS } from '@/types';
import type { JobStatus } from '@/types';
import {
  AudioLines,
  Brain,
  Captions,
  CheckCircle2,
  Crop,
  Film,
  Loader2,
  Scissors,
  XCircle,
} from 'lucide-react';

interface ProcessingProgressProps {
  status: JobStatus;
  progress: number;
  currentStep: string;
  clipsProcessed?: number;
  totalClips?: number;
}

const STEPS = [
  { key: 'extracting_audio', label: 'Extract Audio', icon: AudioLines },
  { key: 'transcribing', label: 'Transcribe', icon: Captions },
  { key: 'analyzing', label: 'Analyze', icon: Brain },
  { key: 'extracting_clips', label: 'Extract', icon: Scissors },
  { key: 'processing_clips', label: 'Process', icon: Crop },
  { key: 'completed', label: 'Complete', icon: CheckCircle2 },
] as const;

export function ProcessingProgress({
  status,
  progress,
  currentStep,
  clipsProcessed = 0,
  totalClips = 0,
}: ProcessingProgressProps) {
  const isFailed = status === 'failed';
  const isCancelled = status === 'cancelled';
  const isCompleted = status === 'completed';
  const isProcessing = !isFailed && !isCancelled && !isCompleted;

  const currentStepIndex = STEPS.findIndex((step) => step.key === status);

  return (
    <div className="space-y-6">
      {/* Status Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isProcessing ? (
            <div className="h-10 w-10 rounded-full bg-blue-500/20 flex items-center justify-center">
              <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
            </div>
          ) : isCompleted ? (
            <div className="h-10 w-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
              <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            </div>
          ) : (
            <div className="h-10 w-10 rounded-full bg-red-500/20 flex items-center justify-center">
              <XCircle className="h-5 w-5 text-red-400" />
            </div>
          )}
          <div>
            <h3 className="font-semibold">
              {isFailed ? 'Processing Failed' : isCancelled ? 'Cancelled' : isCompleted ? 'Complete' : 'Processing'}
            </h3>
            <p className="text-sm text-muted-foreground">{currentStep}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold">{Math.round(progress)}%</p>
          {totalClips > 0 && (
            <p className="text-sm text-muted-foreground">
              {clipsProcessed}/{totalClips} clips
            </p>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="relative">
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-1000 ease-out',
              isFailed ? 'bg-red-500' : isCompleted ? 'bg-emerald-500' : 'gradient-bg animate-gradient'
            )}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Step Indicators */}
      <div className="grid grid-cols-6 gap-2">
        {STEPS.map((step, index) => {
          const isComplete = index < currentStepIndex || isCompleted;
          const isCurrent = step.key === status;
          const StepIcon = step.icon;

          return (
            <div
              key={step.key}
              className={cn(
                'flex flex-col items-center gap-2 p-3 rounded-lg transition-all',
                isComplete && 'bg-emerald-500/10',
                isCurrent && 'bg-primary/10 ring-2 ring-primary/20',
                !isComplete && !isCurrent && 'opacity-50'
              )}
            >
              <div
                className={cn(
                  'h-8 w-8 rounded-full flex items-center justify-center',
                  isComplete
                    ? 'bg-emerald-500/20 text-emerald-400'
                    : isCurrent
                    ? 'bg-primary/20 text-primary'
                    : 'bg-muted text-muted-foreground'
                )}
              >
                {isComplete ? (
                  <CheckCircle2 className="h-4 w-4" />
                ) : (
                  <StepIcon className="h-4 w-4" />
                )}
              </div>
              <span className="text-xs font-medium text-center">
                {step.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
