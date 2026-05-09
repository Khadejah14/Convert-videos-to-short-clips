'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { cn, formatFileSize } from '@/lib/utils';
import { Upload, Film, X, Check, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import type { UploadConfig } from '@/types';
import { CAPTION_STYLES } from '@/types';

interface VideoUploadProps {
  onUpload: (file: File, config: UploadConfig) => void;
  isUploading?: boolean;
}

export function VideoUpload({ onUpload, isUploading }: VideoUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<UploadConfig>({
    clip_count: 3,
    clip_length: 30,
    caption_style: 'default',
    use_vision: false,
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'],
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB
  });

  const handleSubmit = () => {
    if (file) {
      onUpload(file, config);
    }
  };

  return (
    <div className="space-y-6">
      {/* Dropzone */}
      <Card
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed cursor-pointer transition-all duration-200 hover:border-primary/50',
          isDragActive && 'border-primary bg-primary/5',
          file && 'border-emerald-500/50 bg-emerald-500/5'
        )}
      >
        <CardContent className="flex flex-col items-center justify-center py-16">
          <input {...getInputProps()} />
          
          {file ? (
            <>
              <div className="h-16 w-16 rounded-full bg-emerald-500/20 flex items-center justify-center mb-4">
                <Check className="h-8 w-8 text-emerald-400" />
              </div>
              <p className="text-lg font-medium mb-1">{file.name}</p>
              <p className="text-sm text-muted-foreground mb-4">
                {formatFileSize(file.size)}
              </p>
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                }}
              >
                <X className="h-4 w-4 mr-2" />
                Remove
              </Button>
            </>
          ) : (
            <>
              <div
                className={cn(
                  'h-16 w-16 rounded-full flex items-center justify-center mb-4 transition-colors',
                  isDragActive
                    ? 'bg-primary/20'
                    : 'bg-muted'
                )}
              >
                {isDragActive ? (
                  <Upload className="h-8 w-8 text-primary animate-bounce" />
                ) : (
                  <Film className="h-8 w-8 text-muted-foreground" />
                )}
              </div>
              <p className="text-lg font-medium mb-1">
                {isDragActive ? 'Drop your video here' : 'Upload a video'}
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                Drag and drop or click to select
              </p>
              <div className="flex gap-2">
                <Badge variant="secondary">MP4</Badge>
                <Badge variant="secondary">AVI</Badge>
                <Badge variant="secondary">MOV</Badge>
                <Badge variant="secondary">MKV</Badge>
              </div>
              <p className="text-xs text-muted-foreground mt-4">
                Maximum file size: 500MB
              </p>
            </>
          )}
        </CardContent>
      </Card>

      {/* Configuration */}
      <Card>
        <CardContent className="p-6 space-y-6">
          <h3 className="text-lg font-semibold">Processing Settings</h3>

          {/* Clip Count */}
          <div className="space-y-2">
            <Label>Number of Clips</Label>
            <div className="flex gap-2">
              {[1, 2, 3].map((count) => (
                <Button
                  key={count}
                  variant={config.clip_count === count ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setConfig({ ...config, clip_count: count })}
                >
                  {count}
                </Button>
              ))}
            </div>
          </div>

          {/* Clip Length */}
          <div className="space-y-2">
            <Label>Clip Length</Label>
            <div className="flex gap-2">
              {[15, 30, 60].map((length) => (
                <Button
                  key={length}
                  variant={config.clip_length === length ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setConfig({ ...config, clip_length: length })}
                >
                  {length}s
                </Button>
              ))}
            </div>
          </div>

          {/* Caption Style */}
          <div className="space-y-2">
            <Label>Caption Style</Label>
            <div className="grid grid-cols-3 gap-2">
              {CAPTION_STYLES.map((style) => (
                <button
                  key={style.value}
                  onClick={() => setConfig({ ...config, caption_style: style.value })}
                  className={cn(
                    'flex flex-col items-center gap-1 p-3 rounded-lg border transition-all text-center',
                    config.caption_style === style.value
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  )}
                >
                  <span className="text-sm font-medium">{style.label}</span>
                  <span className="text-xs text-muted-foreground">
                    {style.description}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Vision Analysis */}
          <div className="flex items-center justify-between p-4 rounded-lg border">
            <div>
              <p className="font-medium">GPT-4o Vision Analysis</p>
              <p className="text-sm text-muted-foreground">
                Analyze visual hooks for better clip selection
              </p>
            </div>
            <button
              onClick={() => setConfig({ ...config, use_vision: !config.use_vision })}
              className={cn(
                'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none',
                config.use_vision ? 'bg-primary' : 'bg-muted'
              )}
            >
              <span
                className={cn(
                  'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow-lg ring-0 transition duration-200 ease-in-out',
                  config.use_vision ? 'translate-x-5' : 'translate-x-0'
                )}
              />
            </button>
          </div>

          {/* Submit Button */}
          <Button
            variant="gradient"
            size="lg"
            className="w-full"
            disabled={!file || isUploading}
            onClick={handleSubmit}
          >
            {isUploading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Uploading...
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5 mr-2" />
                Generate Clips
              </>
            )}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
