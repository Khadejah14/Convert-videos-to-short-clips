'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  PLATFORM_LABELS,
  PLATFORM_COLORS,
  PUBLISH_STATUS_LABELS,
  type Platform,
  type PublishHistoryEntry,
  type PublishStatus,
} from '@/types';
import {
  ExternalLink,
  RefreshCw,
  Filter,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  Send,
} from 'lucide-react';
import toast from 'react-hot-toast';

const STATUS_COLORS: Record<PublishStatus, string> = {
  uploading: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  processing: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  published: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  failed: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const STATUS_ICONS: Record<PublishStatus, React.ReactNode> = {
  uploading: <Loader2 className="h-4 w-4 animate-spin" />,
  processing: <Loader2 className="h-4 w-4 animate-spin" />,
  published: <CheckCircle2 className="h-4 w-4" />,
  failed: <XCircle className="h-4 w-4" />,
};

export function PublishingHistory() {
  const { publishHistory, setPublishHistory, updateHistoryEntry } = useAppStore();
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [filterPlatform, setFilterPlatform] = useState<Platform | ''>('');
  const perPage = 20;

  useEffect(() => {
    loadHistory();
  }, [page, filterPlatform]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const resp = await api.listPublishHistory(
        filterPlatform || undefined,
        page,
        perPage
      );
      setPublishHistory(resp.history);
      setTotal(resp.total);
    } catch {
      toast.error('Failed to load publishing history');
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = async (historyId: string) => {
    try {
      const updated = await api.retryPublish(historyId);
      updateHistoryEntry(updated);
      toast.success('Retrying publish...');
    } catch {
      toast.error('Failed to retry');
    }
  };

  const totalPages = Math.ceil(total / perPage);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Send className="h-5 w-5" />
            Publishing History
          </CardTitle>
          <div className="flex items-center gap-2">
            <select
              value={filterPlatform}
              onChange={(e) => {
                setFilterPlatform(e.target.value as Platform | '');
                setPage(1);
              }}
              className="h-9 rounded-md border border-input bg-background px-3 text-sm"
            >
              <option value="">All Platforms</option>
              <option value="tiktok">TikTok</option>
              <option value="youtube">YouTube Shorts</option>
              <option value="instagram">Instagram Reels</option>
            </select>
            <Button variant="ghost" size="sm" onClick={loadHistory}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 bg-muted rounded-lg animate-pulse" />
            ))}
          </div>
        ) : publishHistory.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Send className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No publishing history yet</p>
            <p className="text-xs">Published clips will appear here</p>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {publishHistory.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-center justify-between p-4 rounded-lg border"
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div
                      className={`h-10 w-10 rounded-lg flex items-center justify-center ${
                        entry.status === 'published'
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : entry.status === 'failed'
                          ? 'bg-red-500/10 text-red-400'
                          : 'bg-primary/10 text-primary'
                      }`}
                    >
                      {STATUS_ICONS[entry.status]}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="font-medium truncate">
                          {entry.title || 'Untitled'}
                        </p>
                        <Badge variant="secondary" className={PLATFORM_COLORS[entry.platform]}>
                          {PLATFORM_LABELS[entry.platform]}
                        </Badge>
                        <Badge variant="secondary" className={STATUS_COLORS[entry.status]}>
                          {PUBLISH_STATUS_LABELS[entry.status]}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-muted-foreground">
                          {formatDate(entry.published_at || entry.created_at)}
                        </span>
                        {entry.visibility && (
                          <span className="text-xs text-muted-foreground capitalize">
                            &middot; {entry.visibility}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 ml-4">
                    {(entry.status === 'uploading' || entry.status === 'processing') && (
                      <div className="w-24">
                        <Progress value={entry.upload_progress} size="sm" />
                        <p className="text-xs text-muted-foreground text-center mt-1">
                          {Math.round(entry.upload_progress)}%
                        </p>
                      </div>
                    )}

                    {entry.platform_post_url && (
                      <a
                        href={entry.platform_post_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:text-primary/80"
                      >
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}

                    {entry.status === 'failed' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleRetry(entry.id)}
                      >
                        <RefreshCw className="h-3 w-3 mr-1" />
                        Retry
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t">
                <p className="text-sm text-muted-foreground">
                  {total} total entries
                </p>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page - 1)}
                    disabled={page <= 1}
                  >
                    <ChevronLeft className="h-4 w-4" />
                  </Button>
                  <span className="text-sm">
                    Page {page} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page + 1)}
                    disabled={page >= totalPages}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
