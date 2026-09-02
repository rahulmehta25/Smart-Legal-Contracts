import { SAMPLE_DEMO_PATH, SAMPLE_DOCUMENT } from "@/lib/sample-analysis";
import { TextLink } from "@/components/ui/text-link";
import { cn } from "@/lib/utils";

interface SampleDemoCardProps {
  className?: string;
  title?: string;
  description?: string;
}

export function SampleDemoCard({
  className,
  title = "View a sample analysis",
  description = "Open a canned SaaS MSA review with clause quotes, risk ratings, and recommendations. No upload required.",
}: SampleDemoCardProps) {
  return (
    <aside className={cn("border-y border-rule py-6", className)}>
      <p className="eyebrow">Always available</p>
      <h3 className="mt-2 font-serif text-2xl font-medium tracking-tight text-ink">{title}</h3>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-muted">{description}</p>
      <p className="mt-1 text-xs text-ink-muted">{SAMPLE_DOCUMENT.filename}</p>
      <p className="mt-4">
        <TextLink href={SAMPLE_DEMO_PATH}>{title === "View a sample analysis" ? "View Demo" : title}</TextLink>
      </p>
    </aside>
  );
}
