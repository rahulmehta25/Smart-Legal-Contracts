import { Button } from "@/components/ui/button";
import Link from "next/link";

interface EmptyStateProps {
  title: string;
  description: string;
  action?: {
    label: string;
    href?: string;
    onClick?: () => void;
  };
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <div className="border-y border-rule py-12">
      <h3 className="font-serif text-2xl font-medium tracking-tight text-ink">{title}</h3>
      <p className="mt-3 max-w-lg text-sm leading-relaxed text-ink-muted">{description}</p>
      {action ? (
        <p className="mt-5">
          {action.href ? (
            <Button asChild variant="link">
              <Link href={action.href}>{action.label}</Link>
            </Button>
          ) : (
            <Button variant="link" onClick={action.onClick}>
              {action.label}
            </Button>
          )}
        </p>
      ) : null}
    </div>
  );
}
