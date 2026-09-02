import type { ReactNode } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";

export function TextLink({
  href,
  children,
  className,
}: {
  href: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <Link href={href} className={cn("text-link inline-flex items-center gap-1", className)}>
      {children}
    </Link>
  );
}
