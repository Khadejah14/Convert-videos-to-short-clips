'use client';

import { Sparkles } from 'lucide-react';

export function Header() {
  return (
    <header className="sticky top-0 z-30 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        <div className="mr-4 hidden md:flex">
          <a className="mr-6 flex items-center space-x-2" href="/">
            <Sparkles className="h-6 w-6" />
            <span className="hidden font-bold sm:inline-block">
              ClipForge
            </span>
          </a>
        </div>
        <div className="flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <div className="w-full flex-1 md:w-auto md:flex-none">
            {/* Search could go here */}
          </div>
          <nav className="flex items-center">
            <div className="h-8 w-8 rounded-full gradient-bg flex items-center justify-center">
              <span className="text-xs font-medium text-white">U</span>
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
}
