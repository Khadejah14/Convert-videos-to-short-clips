'use client';

import { Toaster } from 'react-hot-toast';

export function ToasterProvider() {
  return (
    <Toaster
      position="bottom-right"
      toastOptions={{
        className: 'bg-card text-card-foreground border',
        duration: 4000,
      }}
    />
  );
}
