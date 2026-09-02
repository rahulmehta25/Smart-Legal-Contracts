"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { TextLink } from "@/components/ui/text-link";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Page error:", error);
  }, [error]);

  return (
    <div className="page-wrap py-24">
      <p className="eyebrow">Error</p>
      <h1 className="display mt-4 text-4xl">Something went wrong.</h1>
      <p className="mt-4 max-w-md text-base leading-relaxed text-ink-muted">
        An unexpected error occurred. Try again, or return home and open the sample analysis.
      </p>
      <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3">
        <Button onClick={reset}>Try again</Button>
        <TextLink href="/">Home</TextLink>
      </div>
    </div>
  );
}
