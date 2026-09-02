import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SAMPLE_DEMO_PATH, SAMPLE_DOCUMENT } from "@/lib/sample-analysis";
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
    <Card className={cn("border-indigo-100 bg-indigo-50/40", className)}>
      <CardContent className="pt-6">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Sparkles className="h-4 w-4 text-indigo-600" />
              <h3 className="text-sm font-medium text-gray-900">{title}</h3>
              <Badge variant="secondary">Always available</Badge>
            </div>
            <p className="text-sm text-gray-600">{description}</p>
            <p className="text-xs text-gray-500 mt-1">{SAMPLE_DOCUMENT.filename}</p>
          </div>
          <Button asChild>
            <Link href={SAMPLE_DEMO_PATH}>
              View Demo
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
