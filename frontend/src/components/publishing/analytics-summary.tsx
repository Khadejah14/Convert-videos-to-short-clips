'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { PLATFORM_LABELS, PLATFORM_COLORS, type Platform, type AnalyticsSummary as AnalyticsSummaryType } from '@/types';
import {
  TrendingUp,
  Eye,
  Heart,
  MessageCircle,
  Share2,
  Clock,
  BarChart3,
} from 'lucide-react';
import toast from 'react-hot-toast';

export function AnalyticsSummaryCard() {
  const { analyticsSummary, setAnalyticsSummary } = useAppStore();
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);

  useEffect(() => {
    loadSummary();
  }, [days]);

  const loadSummary = async () => {
    try {
      const summary = await api.getAnalyticsSummary(undefined, days);
      setAnalyticsSummary(summary);
    } catch {
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (n: number) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
  };

  const formatWatchTime = (seconds: number) => {
    if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
    if (seconds >= 60) return `${(seconds / 60).toFixed(0)}m`;
    return `${Math.round(seconds)}s`;
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-20 bg-muted rounded-lg animate-pulse" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!analyticsSummary || analyticsSummary.posts_count === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Creator Analytics
          </CardTitle>
          <CardDescription>
            Performance metrics across all platforms
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <TrendingUp className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No analytics data yet</p>
            <p className="text-xs">Publish clips to start seeing metrics</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Creator Analytics
            </CardTitle>
            <CardDescription>
              Performance across {analyticsSummary.posts_count} published clips
            </CardDescription>
          </div>
          <select
            value={days}
            onChange={(e) => setDays(parseInt(e.target.value))}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={365}>Last year</option>
          </select>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Eye className="h-4 w-4" />
              <span className="text-xs">Views</span>
            </div>
            <p className="text-2xl font-bold">{formatNumber(analyticsSummary.total_views)}</p>
          </div>
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Heart className="h-4 w-4" />
              <span className="text-xs">Likes</span>
            </div>
            <p className="text-2xl font-bold">{formatNumber(analyticsSummary.total_likes)}</p>
          </div>
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <MessageCircle className="h-4 w-4" />
              <span className="text-xs">Comments</span>
            </div>
            <p className="text-2xl font-bold">{formatNumber(analyticsSummary.total_comments)}</p>
          </div>
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Share2 className="h-4 w-4" />
              <span className="text-xs">Shares</span>
            </div>
            <p className="text-2xl font-bold">{formatNumber(analyticsSummary.total_shares)}</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Clock className="h-4 w-4" />
              <span className="text-xs">Total Watch Time</span>
            </div>
            <p className="text-xl font-bold">
              {formatWatchTime(analyticsSummary.total_watch_time)}
            </p>
          </div>
          <div className="p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <TrendingUp className="h-4 w-4" />
              <span className="text-xs">Avg Engagement Rate</span>
            </div>
            <p className="text-xl font-bold">
              {analyticsSummary.average_engagement_rate.toFixed(1)}%
            </p>
          </div>
        </div>

        {Object.keys(analyticsSummary.platform_breakdown).length > 0 && (
          <div>
            <p className="text-sm font-medium mb-3">Posts by Platform</p>
            <div className="flex gap-2">
              {Object.entries(analyticsSummary.platform_breakdown).map(
                ([platform, count]) => (
                  <Badge
                    key={platform}
                    variant="secondary"
                    className={PLATFORM_COLORS[platform as Platform]}
                  >
                    {PLATFORM_LABELS[platform as Platform]}: {count}
                  </Badge>
                )
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
