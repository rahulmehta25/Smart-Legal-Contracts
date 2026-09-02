"use client";

import { useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  formatClauseType,
  formatConfidence,
  getRiskBadgeVariant,
} from "@/lib/utils";
import type { ArbitrationClause } from "@/types/api";

export function ClauseCard({ clause, index }: { clause: ArbitrationClause; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);
  const riskLevel = clause.risk_level || "medium";

  return (
    <article className="border-t border-rule last:border-b">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start justify-between gap-6 py-6 text-left hover-short"
        aria-expanded={expanded}
      >
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="font-serif text-xl font-medium tracking-tight text-ink">
              {formatClauseType(clause.clause_type)}
            </h3>
            <Badge variant={getRiskBadgeVariant(riskLevel)}>{riskLevel} risk</Badge>
          </div>
          {clause.section_reference ? (
            <p className="mt-1 text-sm text-ink-muted">{clause.section_reference}</p>
          ) : null}
        </div>
        <div className="shrink-0 text-right">
          <p className="text-sm text-ink-muted">{formatConfidence(clause.confidence_score)}</p>
          <p className="mt-1 text-xs uppercase tracking-[0.14em] text-ink-muted">
            {expanded ? "Hide" : "Show"}
          </p>
        </div>
      </button>

      {expanded && (
        <div className="space-y-5 pb-8">
          <blockquote className="border-l-2 border-brass pl-5">
            <p className="font-serif text-base leading-relaxed text-ink">
              &quot;{clause.clause_text}&quot;
            </p>
          </blockquote>

          {clause.impact_summary ? (
            <div>
              <p className="eyebrow">Impact</p>
              <p className="mt-2 text-sm leading-relaxed text-ink-muted">{clause.impact_summary}</p>
            </div>
          ) : null}

          {clause.recommendations && clause.recommendations.length > 0 ? (
            <div>
              <p className="eyebrow">Recommendations</p>
              <ul className="mt-3 space-y-2">
                {clause.recommendations.map((rec) => (
                  <li key={rec} className="text-sm leading-relaxed text-ink-muted">
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <Button
            variant="link"
            size="sm"
            onClick={() => {
              void navigator.clipboard?.writeText(clause.clause_text);
              toast.success("Copied to clipboard");
            }}
          >
            Copy clause
          </Button>
        </div>
      )}
    </article>
  );
}
