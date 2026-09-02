"use client";

import type { CSSProperties, ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const style: CSSProperties | undefined = delay
    ? { animationDelay: `${delay}s` }
    : undefined;

  return (
    <div data-reveal="" className={cn("animate-fade-rise", className)} style={style}>
      {children}
    </div>
  );
}
