import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Sample Analysis | Smart Legal Contracts",
  description:
    "Review a canned SaaS MSA analysis with arbitration, jury waiver, and class-action findings. No upload required.",
};

export default function DemoLayout({ children }: { children: ReactNode }) {
  return children;
}
