import type { ArbitrationAnalysis, ArbitrationClause, Document } from "../types/api";

export const SAMPLE_ANALYSIS_SLUG = "sample";
export const SAMPLE_ANALYSIS_ID = 900001;
export const SAMPLE_DOCUMENT_ID = 900001;
export const SAMPLE_DEMO_PATH = "/demo";

const SAMPLE_ANALYZED_AT = "2026-03-12T15:42:00.000Z";

const sampleClauses: ArbitrationClause[] = [
  {
    id: 900101,
    analysis_id: SAMPLE_ANALYSIS_ID,
    clause_type: "mandatory_arbitration",
    clause_text:
      "Any dispute, claim, or controversy arising out of or relating to this Agreement, or the breach, termination, enforcement, interpretation, or validity thereof, shall be resolved by binding arbitration administered by the American Arbitration Association in accordance with its Commercial Arbitration Rules.",
    confidence_score: 0.94,
    risk_level: "high",
    section_reference: "Section 12.1 Dispute Resolution",
    start_position: 18420,
    end_position: 18810,
    impact_summary:
      "The customer is required to arbitrate most claims and cannot take those disputes to a public court in the first instance.",
    recommendations: [
      "Confirm whether consumer or employment claims should be carved out.",
      "Ask for a mutual carve-out for injunctive relief to protect IP.",
      "Document the filing fees and who pays them before signature.",
    ],
  },
  {
    id: 900102,
    analysis_id: SAMPLE_ANALYSIS_ID,
    clause_type: "jury_waiver",
    clause_text:
      "TO THE FULLEST EXTENT PERMITTED BY LAW, EACH PARTY WAIVES ANY RIGHT TO A JURY TRIAL IN ANY PROCEEDING ARISING OUT OF OR RELATED TO THIS AGREEMENT.",
    confidence_score: 0.91,
    risk_level: "high",
    section_reference: "Section 12.2 Jury Waiver",
    start_position: 18820,
    end_position: 19005,
    impact_summary:
      "If a claim leaves arbitration, the customer still gives up a jury and proceeds before a judge only.",
    recommendations: [
      "Keep this waiver only if it is mutual and conspicuous.",
      "Check whether local law limits jury waivers in consumer contracts.",
    ],
  },
  {
    id: 900103,
    analysis_id: SAMPLE_ANALYSIS_ID,
    clause_type: "class_action_waiver",
    clause_text:
      "You agree to bring claims only in your individual capacity and not as a plaintiff or class member in any purported class, collective, or representative proceeding. The arbitrator may not consolidate more than one person's claims.",
    confidence_score: 0.89,
    risk_level: "medium",
    section_reference: "Section 12.3 Class Waiver",
    start_position: 19020,
    end_position: 19340,
    impact_summary:
      "Low-dollar claims become harder to pursue because they cannot be grouped with other customers.",
    recommendations: [
      "Request a small-claims court carve-out for invoices under a set amount.",
      "Confirm the waiver is mutual so the vendor cannot file class claims either.",
    ],
  },
  {
    id: 900104,
    analysis_id: SAMPLE_ANALYSIS_ID,
    clause_type: "forum_selection",
    clause_text:
      "The seat of arbitration shall be Wilmington, Delaware. The language of the arbitration shall be English. Judgment on the award may be entered in any court of competent jurisdiction.",
    confidence_score: 0.86,
    risk_level: "medium",
    section_reference: "Section 12.4 Seat and Language",
    start_position: 19350,
    end_position: 19580,
    impact_summary:
      "Travel and local counsel costs rise if the customer is outside Delaware.",
    recommendations: [
      "Negotiate a seat closer to the customer's principal place of business.",
      "Keep remote hearings as the default to limit travel.",
    ],
  },
  {
    id: 900105,
    analysis_id: SAMPLE_ANALYSIS_ID,
    clause_type: "mediation_first",
    clause_text:
      "Before commencing arbitration, the parties shall first attempt to resolve the dispute through good-faith mediation for a period of thirty (30) days, unless a party seeks interim injunctive relief.",
    confidence_score: 0.82,
    risk_level: "low",
    section_reference: "Section 12.0 Escalation",
    start_position: 18110,
    end_position: 18390,
    impact_summary:
      "A short mediation step can settle billing or SLA disputes before formal arbitration starts.",
    recommendations: [
      "Keep the 30-day window, and name a default mediation provider.",
      "Preserve the injunction carve-out for IP and data incidents.",
    ],
  },
];

export const SAMPLE_DOCUMENT: Document = {
  id: SAMPLE_DOCUMENT_ID,
  filename: "Acme_SaaS_MSA_2024.pdf",
  content: "",
  content_type: "application/pdf",
  file_size: 248320,
  page_count: 18,
  processed: true,
  created_at: SAMPLE_ANALYZED_AT,
  updated_at: SAMPLE_ANALYZED_AT,
};

export const SAMPLE_ANALYSIS: ArbitrationAnalysis = {
  id: SAMPLE_ANALYSIS_ID,
  document_id: SAMPLE_DOCUMENT_ID,
  has_arbitration_clause: true,
  confidence_score: 0.92,
  analysis_summary:
    "This SaaS master services agreement requires binding AAA arbitration in Delaware, waives jury trial, and bars class actions. A 30-day mediation step comes first, with a narrow injunction carve-out. Overall risk is high for a customer that wants public-court remedies or collective claims.",
  analyzed_at: SAMPLE_ANALYZED_AT,
  analysis_version: "sample-2.0.0",
  processing_time_ms: 1840,
  clauses: sampleClauses,
  risk_level: "high",
};

export function isSampleAnalysisId(id: string | number | null | undefined): boolean {
  if (id === null || id === undefined) return false;
  const normalized = String(id).trim().toLowerCase();
  return (
    normalized === SAMPLE_ANALYSIS_SLUG ||
    normalized === String(SAMPLE_ANALYSIS_ID) ||
    normalized === String(SAMPLE_DOCUMENT_ID)
  );
}

export function getSampleAnalysis(): ArbitrationAnalysis {
  return SAMPLE_ANALYSIS;
}

export function getSampleDocument(): Document {
  return SAMPLE_DOCUMENT;
}

export function resolveSampleAnalysis(
  id: string | number | null | undefined
): ArbitrationAnalysis | null {
  return isSampleAnalysisId(id) ? SAMPLE_ANALYSIS : null;
}

export function resolveSampleDocument(
  id: string | number | null | undefined
): Document | null {
  return isSampleAnalysisId(id) ? SAMPLE_DOCUMENT : null;
}
