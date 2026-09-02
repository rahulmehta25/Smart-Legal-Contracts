import assert from "node:assert/strict";
import {
  SAMPLE_ANALYSIS,
  SAMPLE_ANALYSIS_ID,
  SAMPLE_ANALYSIS_SLUG,
  SAMPLE_DEMO_PATH,
  SAMPLE_DOCUMENT,
  getSampleAnalysis,
  getSampleDocument,
  isSampleAnalysisId,
  resolveSampleAnalysis,
  resolveSampleDocument,
} from "./sample-analysis";

function testSampleAnalysisIsSelfContained() {
  const analysis = getSampleAnalysis();
  const document = getSampleDocument();

  assert.equal(analysis.id, SAMPLE_ANALYSIS_ID);
  assert.equal(document.id, SAMPLE_DOCUMENT.id);
  assert.equal(analysis.document_id, document.id);
  assert.equal(analysis.has_arbitration_clause, true);
  assert.ok(analysis.clauses.length >= 4);
  assert.ok(analysis.confidence_score > 0.8);
  assert.ok(analysis.analysis_summary.length > 40);
  assert.ok(analysis.clauses.some((clause) => clause.risk_level === "high"));

  for (const clause of analysis.clauses) {
    assert.ok(clause.clause_text.length > 20);
    assert.ok(clause.confidence_score > 0 && clause.confidence_score <= 1);
    assert.ok(["high", "medium", "low"].includes(clause.risk_level));
    assert.ok((clause.recommendations?.length ?? 0) > 0);
    assert.doesNotMatch(clause.clause_text, /\u2013|\u2014/);
    assert.doesNotMatch(clause.impact_summary ?? "", /\u2013|\u2014/);
  }

  assert.doesNotMatch(analysis.analysis_summary, /\u2013|\u2014/);
  assert.doesNotMatch(document.filename, /\u2013|\u2014/);
}

function testSampleIdsResolveWithoutApi() {
  assert.equal(SAMPLE_DEMO_PATH, "/demo");
  assert.equal(isSampleAnalysisId("sample"), true);
  assert.equal(isSampleAnalysisId("SAMPLE"), true);
  assert.equal(isSampleAnalysisId(SAMPLE_ANALYSIS_SLUG), true);
  assert.equal(isSampleAnalysisId(SAMPLE_ANALYSIS_ID), true);
  assert.equal(isSampleAnalysisId(String(SAMPLE_ANALYSIS_ID)), true);
  assert.equal(isSampleAnalysisId("  sample  "), true);
  assert.equal(isSampleAnalysisId(12), false);
  assert.equal(isSampleAnalysisId("live"), false);
  assert.equal(isSampleAnalysisId(null), false);
  assert.equal(isSampleAnalysisId(undefined), false);
  assert.equal(resolveSampleAnalysis("sample"), SAMPLE_ANALYSIS);
  assert.equal(resolveSampleDocument("sample"), SAMPLE_DOCUMENT);
  assert.equal(resolveSampleAnalysis("missing"), null);
}

testSampleAnalysisIsSelfContained();
testSampleIdsResolveWithoutApi();

console.log("sample-analysis tests passed");
