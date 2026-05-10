'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAppStore } from '@/lib/store';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  PLATFORM_LABELS,
  PLATFORM_COLORS,
  type Platform,
  type ConnectedAccount,
} from '@/types';
import {
  Link2,
  Unlink,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Youtube,
  Instagram,
} from 'lucide-react';
import toast from 'react-hot-toast';

const PLATFORM_ICONS: Record<Platform, React.ReactNode> = {
  tiktok: (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-2.88 2.5 2.89 2.89 0 01-2.89-2.89 2.89 2.89 0 012.89-2.89c.28 0 .54.04.79.1v-3.5a6.37 6.37 0 00-.79-.05A6.34 6.34 0 003.15 15.2a6.34 6.34 0 006.34 6.34 6.34 6.34 0 006.34-6.34V8.74a8.27 8.27 0 004.76 1.52V6.81a4.84 4.84 0 01-1-.12z" />
    </svg>
  ),
  youtube: <Youtube className="h-5 w-5" />,
  instagram: <Instagram className="h-5 w-5" />,
};

const PLATFORM_DESCRIPTIONS: Record<Platform, string> = {
  tiktok: 'Upload short-form videos directly to TikTok',
  youtube: 'Publish to YouTube Shorts',
  instagram: 'Share as Instagram Reels',
};

interface Props {
  onAccountSelect?: (account: ConnectedAccount) => void;
  selectable?: boolean;
  selectedAccountId?: string;
}

export function ConnectedAccounts({ onAccountSelect, selectable, selectedAccountId }: Props) {
  const {
    connectedAccounts,
    setConnectedAccounts,
    addConnectedAccount,
    removeConnectedAccount,
  } = useAppStore();
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState<Platform | null>(null);

  useEffect(() => {
    loadAccounts();
  }, []);

  const loadAccounts = async () => {
    try {
      const resp = await api.listConnectedAccounts();
      setConnectedAccounts(resp.accounts);
    } catch {
      toast.error('Failed to load connected accounts');
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (platform: Platform) => {
    setConnecting(platform);
    try {
      const { authorization_url } = await api.getOAuthUrl(platform);
      const width = 600;
      const height = 700;
      const left = window.screenX + (window.outerWidth - width) / 2;
      const top = window.screenY + (window.outerHeight - height) / 2;

      const popup = window.open(
        authorization_url,
        `oauth_${platform}`,
        `width=${width},height=${height},left=${left},top=${top},popup=yes`
      );

      const checkClosed = setInterval(() => {
        if (popup?.closed) {
          clearInterval(checkClosed);
          setConnecting(null);
          loadAccounts();
        }
      }, 500);
    } catch {
      toast.error('Failed to initiate connection');
      setConnecting(null);
    }
  };

  const handleDisconnect = async (accountId: string) => {
    try {
      await api.disconnectAccount(accountId);
      removeConnectedAccount(accountId);
      toast.success('Account disconnected');
    } catch {
      toast.error('Failed to disconnect account');
    }
  };

  const getAccountsForPlatform = (platform: Platform) =>
    connectedAccounts.filter((a) => a.platform === platform);

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-muted rounded-lg" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {(['tiktok', 'youtube', 'instagram'] as Platform[]).map((platform) => {
        const accounts = getAccountsForPlatform(platform);
        const isConnected = accounts.length > 0;
        const isConnecting = connecting === platform;

        return (
          <Card key={platform}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className={`h-10 w-10 rounded-lg flex items-center justify-center ${
                      isConnected
                        ? 'bg-primary/10 text-primary'
                        : 'bg-muted text-muted-foreground'
                    }`}
                  >
                    {PLATFORM_ICONS[platform]}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="font-medium">{PLATFORM_LABELS[platform]}</p>
                      {isConnected ? (
                        <Badge variant="secondary" className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                          <CheckCircle2 className="h-3 w-3 mr-1" />
                          Connected
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="bg-muted text-muted-foreground">
                          Not connected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {PLATFORM_DESCRIPTIONS[platform]}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isConnected ? (
                    <>
                      {selectable && (
                        <Button
                          variant={selectedAccountId === accounts[0].id ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => onAccountSelect?.(accounts[0])}
                        >
                          {selectedAccountId === accounts[0].id ? 'Selected' : 'Select'}
                        </Button>
                      )}
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDisconnect(accounts[0].id)}
                        className="text-destructive hover:text-destructive"
                      >
                        <Unlink className="h-4 w-4 mr-1" />
                        Disconnect
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleConnect(platform)}
                      disabled={isConnecting}
                    >
                      {isConnecting ? (
                        'Connecting...'
                      ) : (
                        <>
                          <Link2 className="h-4 w-4 mr-1" />
                          Connect
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </div>

              {isConnected && accounts[0] && (
                <div className="mt-3 flex items-center gap-3 pl-[52px]">
                  {accounts[0].avatar_url && (
                    <img
                      src={accounts[0].avatar_url}
                      alt=""
                      className="h-8 w-8 rounded-full"
                    />
                  )}
                  <div>
                    <p className="text-sm font-medium">
                      {accounts[0].display_name || accounts[0].platform_username}
                    </p>
                    {accounts[0].platform_username && accounts[0].display_name && (
                      <p className="text-xs text-muted-foreground">
                        @{accounts[0].platform_username}
                      </p>
                    )}
                  </div>
                  {accounts[0].status === 'expired' && (
                    <Badge variant="secondary" className="bg-amber-500/20 text-amber-400 border-amber-500/30 ml-auto">
                      <AlertCircle className="h-3 w-3 mr-1" />
                      Token expired
                    </Badge>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
