'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  PLATFORM_LABELS,
  type ScheduleStatus,
  type PublishSchedule,
} from '@/types';
import {
  Clock,
  XCircle,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Ban,
} from 'lucide-react';
import toast from 'react-hot-toast';

const SCHEDULE_STATUS_COLORS: Record<ScheduleStatus, string> = {
  pending: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  processing: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  completed: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  failed: 'bg-red-500/20 text-red-400 border-red-500/30',
  cancelled: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
};

export function ScheduledPublishes() {
  const { schedules, setSchedules, removeSchedule } = useAppStore();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSchedules();
  }, []);

  const loadSchedules = async () => {
    try {
      const resp = await api.listSchedules();
      setSchedules(resp.schedules);
    } catch {
      toast.error('Failed to load schedules');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (scheduleId: string) => {
    try {
      await api.cancelSchedule(scheduleId);
      removeSchedule(scheduleId);
      toast.success('Schedule cancelled');
    } catch {
      toast.error('Failed to cancel schedule');
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const pendingSchedules = schedules.filter((s) => s.status === 'pending');

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="space-y-3">
            {[1, 2].map((i) => (
              <div key={i} className="h-16 bg-muted rounded-lg animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (pendingSchedules.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Scheduled Publishes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No scheduled publishes</p>
            <p className="text-xs">Schedule a publish from the publish dialog</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Clock className="h-5 w-5" />
          Scheduled Publishes
          <Badge variant="secondary" className="ml-auto">
            {pendingSchedules.length}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {pendingSchedules.map((schedule) => (
            <div
              key={schedule.id}
              className="flex items-center justify-between p-3 rounded-lg border"
            >
              <div className="flex items-center gap-3">
                <div className="h-8 w-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                  <Clock className="h-4 w-4 text-blue-400" />
                </div>
                <div>
                  <p className="text-sm font-medium">
                    {formatDate(schedule.scheduled_at)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Created {formatDate(schedule.created_at)}
                  </p>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleCancel(schedule.id)}
                className="text-destructive hover:text-destructive"
              >
                <Ban className="h-4 w-4 mr-1" />
                Cancel
              </Button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
