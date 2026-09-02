"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { AlertTriangle, CheckCircle, ChevronDown, ChevronUp, Copy } from "lucide-react";
import {
  cn,
  copyToClipboard,
  formatClauseType,
  formatConfidence,
  getRiskBadgeVariant,
  getRiskBgColor,
  getRiskBorderColor,
  getRiskColor,
} from "@/lib/utils";
import type { ArbitrationClause } from "@/types/api";

export function ClauseCard({ clause, index }: { clause: ArbitrationClause; index: number }) {
  const [expanded, setExpanded] = useState(index === 0);
  const riskLevel = clause.risk_level || "medium";

  return (
    <Card className={cn("border", getRiskBorderColor(riskLevel))}>
      <button onClick={() => setExpanded(!expanded)} className="w-full text-left">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <Badge variant={getRiskBadgeVariant(riskLevel)}>{riskLevel.toUpperCase()}</Badge>
              </motion.div>
              <div>
                <CardTitle className="text-base font-medium">
                  {formatClauseType(clause.clause_type)}
                </CardTitle>
                {clause.section_reference && (
                  <CardDescription className="text-xs mt-0.5">
                    {clause.section_reference}
                  </CardDescription>
                )}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  {formatConfidence(clause.confidence_score)}
                </span>
                <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-blue-500 rounded-full"
                    initial={{ width: 0 }}
                    animate={{ width: `${clause.confidence_score * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.3 + index * 0.1, ease: "easeOut" }}
                  />
                </div>
              </div>
              {expanded ? (
                <ChevronUp className="h-4 w-4 text-gray-400" />
              ) : (
                <ChevronDown className="h-4 w-4 text-gray-400" />
              )}
            </div>
          </div>
        </CardHeader>
      </button>

      {expanded && (
        <>
          <Separator />
          <CardContent className="pt-4 space-y-4">
            <div
              className={cn(
                "p-4 rounded-lg border-l-4",
                getRiskBgColor(riskLevel),
                getRiskBorderColor(riskLevel).replace("border-", "border-l-")
              )}
            >
              <p className="text-sm text-gray-700 italic leading-relaxed">
                &quot;{clause.clause_text}&quot;
              </p>
            </div>

            {clause.impact_summary && (
              <div className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                <AlertTriangle
                  className={cn("h-4 w-4 mt-0.5 flex-shrink-0", getRiskColor(riskLevel))}
                />
                <div>
                  <p className="text-xs font-medium text-gray-700 mb-0.5">Impact Assessment</p>
                  <p className="text-xs text-gray-600">{clause.impact_summary}</p>
                </div>
              </div>
            )}

            {clause.recommendations && clause.recommendations.length > 0 && (
              <div>
                <p className="text-xs font-medium text-gray-700 mb-2">Recommendations</p>
                <ul className="space-y-1">
                  {clause.recommendations.map((rec) => (
                    <li key={rec} className="flex items-start gap-2 text-xs text-gray-600">
                      <CheckCircle className="h-3 w-3 text-emerald-500 mt-0.5 flex-shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  copyToClipboard(clause.clause_text);
                  toast.success("Copied to clipboard");
                }}
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>
            </div>
          </CardContent>
        </>
      )}
    </Card>
  );
}
