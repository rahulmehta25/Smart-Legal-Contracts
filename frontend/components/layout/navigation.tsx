"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { SAMPLE_DEMO_PATH } from "@/lib/sample-analysis";
import { Menu, X } from "lucide-react";

const navigation = [
  { name: "Demo", href: SAMPLE_DEMO_PATH },
  { name: "Upload", href: "/upload" },
  { name: "History", href: "/history" },
  { name: "Batch", href: "/batch" },
  { name: "Compare", href: "/compare" },
  { name: "Settings", href: "/settings" },
];

export function Navigation() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-rule bg-ivory/90 backdrop-blur-md">
      <div className="page-wrap">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="group flex items-baseline gap-2.5 hover-short">
            <span className="font-serif text-[1.35rem] font-medium tracking-tight text-ink">
              Smart Legal
            </span>
            <span className="hidden text-[0.7rem] uppercase tracking-[0.18em] text-ink-muted sm:inline">
              Contracts
            </span>
          </Link>

          <nav className="hidden items-center gap-7 md:flex">
            {navigation.map((item) => {
              const isActive =
                pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "text-sm tracking-wide hover-short",
                    isActive
                      ? "text-ink underline decoration-brass decoration-1 underline-offset-[6px]"
                      : "text-ink-muted hover:text-ink"
                  )}
                >
                  {item.name}
                </Link>
              );
            })}
          </nav>

          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center text-ink md:hidden"
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileOpen}
            onClick={() => setMobileOpen((open) => !open)}
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>

        {mobileOpen && (
          <nav className="space-y-1 border-t border-rule pb-4 pt-2 md:hidden">
            {navigation.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={cn(
                    "block px-1 py-2 text-sm",
                    isActive ? "text-ink underline decoration-brass underline-offset-4" : "text-ink-muted"
                  )}
                >
                  {item.name}
                </Link>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
}
