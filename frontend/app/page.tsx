"use client";

import { useRef } from "react";
import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  FileText,
  Upload,
  Shield,
  Zap,
  ArrowRight,
  Clock,
  BarChart3,
  Search,
  FileSearch,
  AlertTriangle,
  Sparkles,
  CheckCircle2,
} from "lucide-react";
import { CountUp } from "@/components/ui/motion";
import { SAMPLE_DEMO_PATH, SAMPLE_DOCUMENT } from "@/lib/sample-analysis";

const features = [
  {
    icon: Search,
    title: "Clause Detection",
    description:
      "Automatically identify arbitration clauses, jury waivers, and dispute resolution provisions.",
  },
  {
    icon: Shield,
    title: "Risk Assessment",
    description:
      "Get instant risk level ratings with detailed impact analysis for each detected clause.",
  },
  {
    icon: Zap,
    title: "Fast Analysis",
    description:
      "Process documents in seconds using our optimized RAG pipeline with 85%+ accuracy.",
  },
  {
    icon: BarChart3,
    title: "Batch Processing",
    description:
      "Analyze multiple documents simultaneously with progress tracking and aggregate results.",
  },
  {
    icon: FileSearch,
    title: "Document Comparison",
    description: "Compare clauses across documents to identify differences and similarities.",
  },
  {
    icon: Clock,
    title: "Analysis History",
    description: "Track all your analyses with filtering, search, and export capabilities.",
  },
];

const stats = [
  { value: 85, label: "Detection Accuracy", prefix: "", suffix: "%+" },
  { value: 2, label: "Analysis Time", prefix: "<", suffix: "s" },
  { value: 50, label: "Max File Size", prefix: "", suffix: "MB" },
  { value: null, label: "Supported Formats", display: "PDF/DOCX" },
];

const clauseTypes = [
  { name: "Mandatory Arbitration", risk: "high" },
  { name: "Jury Trial Waiver", risk: "high" },
  { name: "Class Action Waiver", risk: "medium" },
  { name: "Forum Selection", risk: "medium" },
  { name: "Mediation First", risk: "low" },
  { name: "Escalation Clauses", risk: "low" },
];

const steps = [
  {
    step: "01",
    title: "Upload a contract",
    description: "Drop a PDF or DOCX. Live upload is optional for this portfolio demo.",
  },
  {
    step: "02",
    title: "Detect dispute language",
    description: "The model flags arbitration, jury waivers, class waivers, and forum terms.",
  },
  {
    step: "03",
    title: "Review risk and next steps",
    description: "Open clause quotes, confidence scores, and recommended negotiation points.",
  },
];

const springTransition = { type: "spring" as const, stiffness: 400, damping: 17 };

export default function HomePage() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  });
  const bgY = useTransform(scrollYProgress, [0, 1], ["0%", "30%"]);

  return (
    <div className="flex flex-col">
      <section
        ref={heroRef}
        className="relative border-b border-gray-100 bg-white overflow-hidden"
      >
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-blue-50/80 via-white to-indigo-50/50"
          style={{ y: bgY }}
        />
        <div className="hero-grid absolute inset-0 opacity-70" />
        <div className="hero-orb hero-orb-left" />
        <div className="hero-orb hero-orb-right" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            <div className="max-w-xl">
              <Badge variant="secondary" className="mb-4 animate-fade-in-up">
                AI contract review for portfolio demos
              </Badge>
              <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mb-4 gradient-text animate-fade-in-up stagger-1">
                Detect arbitration clauses in seconds
              </h1>
              <p className="text-lg text-gray-600 mb-8 leading-relaxed animate-fade-in-up stagger-2">
                Upload a legal document or open the sample analysis to see risk ratings,
                clause quotes, and recommended next steps. The sample path works even
                when live upload is unavailable.
              </p>
              <div className="flex flex-wrap gap-3 animate-fade-in-up stagger-3">
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  transition={springTransition}
                >
                  <Button asChild size="lg">
                    <Link href={SAMPLE_DEMO_PATH}>
                      <Sparkles className="mr-2 h-4 w-4" />
                      View Demo
                    </Link>
                  </Button>
                </motion.div>
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  transition={springTransition}
                >
                  <Button variant="outline" size="lg" asChild>
                    <Link href="/upload">
                      <Upload className="mr-2 h-4 w-4" />
                      Upload Document
                    </Link>
                  </Button>
                </motion.div>
              </div>
              <p className="mt-4 text-sm text-gray-500 animate-fade-in-up stagger-4">
                Sample file: {SAMPLE_DOCUMENT.filename}
              </p>
            </div>

            <Link
              href={SAMPLE_DEMO_PATH}
              className="block animate-fade-in-right stagger-3 group"
              aria-label="Open sample analysis"
            >
              <div className="relative">
                <div className="absolute -inset-3 rounded-2xl bg-gradient-to-br from-blue-200/40 to-indigo-200/30 blur-xl opacity-70 group-hover:opacity-100 transition-opacity" />
                <Card className="relative border-gray-200/80 shadow-xl shadow-indigo-100/60 overflow-hidden">
                  <CardContent className="p-5 sm:p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-gray-400" />
                        <span className="text-sm font-medium text-gray-800">
                          {SAMPLE_DOCUMENT.filename}
                        </span>
                      </div>
                      <Badge variant="danger">High risk</Badge>
                    </div>
                    <div className="space-y-3">
                      <div className="p-3.5 bg-red-50 border border-red-100 rounded-lg">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                          <div>
                            <div className="font-medium text-red-800 text-sm mb-1">
                              Mandatory Arbitration
                            </div>
                            <p className="text-xs text-red-700 leading-relaxed">
                              Binding AAA arbitration for disputes arising out of this
                              Agreement.
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="danger">High Risk</Badge>
                              <span className="text-xs text-red-600">94% confidence</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      <div className="p-3.5 bg-amber-50 border border-amber-100 rounded-lg">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0 mt-0.5" />
                          <div>
                            <div className="font-medium text-amber-800 text-sm mb-1">
                              Class Action Waiver
                            </div>
                            <p className="text-xs text-amber-700 leading-relaxed">
                              Claims may be brought only on an individual basis.
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge variant="warning">Medium Risk</Badge>
                              <span className="text-xs text-amber-600">89% confidence</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="mt-4 flex items-center justify-between text-sm text-indigo-700 font-medium">
                      <span>Open full sample analysis</span>
                      <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </Link>
          </div>
        </div>
      </section>

      <section className="border-b border-gray-100 bg-gray-50/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, i) => (
              <div
                key={stat.label}
                className={`text-center animate-fade-in-up stagger-${i + 1}`}
              >
                <div className="text-3xl font-semibold text-gray-900">
                  {stat.value !== null ? (
                    <CountUp end={stat.value} prefix={stat.prefix} suffix={stat.suffix} />
                  ) : (
                    stat.display
                  )}
                </div>
                <div className="text-sm text-gray-500 mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b border-gray-100 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="text-center mb-12 animate-fade-in-up">
            <h2 className="text-2xl font-semibold text-gray-900 mb-3">
              How the review works
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Three steps from document to a readable risk memo. Prefer a guaranteed
              walkthrough? Use View Demo.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {steps.map((item, i) => (
              <div key={item.step} className={`animate-fade-in-up stagger-${i + 1}`}>
                <Card className="h-full border-gray-200">
                  <CardContent className="pt-6">
                    <div className="text-xs font-semibold tracking-wider text-indigo-600 mb-3">
                      {item.step}
                    </div>
                    <h3 className="font-medium text-gray-900 mb-2">{item.title}</h3>
                    <p className="text-sm text-gray-500 leading-relaxed">{item.description}</p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="border-b border-gray-100 bg-gray-50/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="text-center mb-12 animate-fade-in-up">
            <h2 className="text-2xl font-semibold text-gray-900 mb-3">
              Everything you need to analyze legal documents
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              The platform combines advanced NLP with retrieval-augmented generation to
              deliver accurate, fast, and actionable insights.
            </p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => {
              const Icon = feature.icon;
              return (
                <div
                  key={feature.title}
                  className={`animate-fade-in-up hover-lift stagger-${i + 1}`}
                >
                  <Card className="border-gray-200 h-full">
                    <CardContent className="pt-6">
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 h-10 w-10 rounded-lg bg-blue-50 flex items-center justify-center">
                          <Icon className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <h3 className="font-medium text-gray-900 mb-1">{feature.title}</h3>
                          <p className="text-sm text-gray-500 leading-relaxed">
                            {feature.description}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="border-b border-gray-100 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="animate-fade-in-left stagger-2">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Comprehensive clause detection
              </h2>
              <p className="text-gray-600 mb-6 leading-relaxed">
                The model is trained to identify arbitration and dispute resolution
                clauses commonly found in contracts and commercial agreements.
              </p>
              <div className="space-y-3">
                {clauseTypes.map((clause) => (
                  <div
                    key={clause.name}
                    className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                  >
                    <span className="text-sm text-gray-700">{clause.name}</span>
                    <Badge
                      variant={
                        clause.risk === "high"
                          ? "danger"
                          : clause.risk === "medium"
                            ? "warning"
                            : "success"
                      }
                    >
                      {clause.risk} risk
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
            <Link
              href={SAMPLE_DEMO_PATH}
              className="block bg-gray-50 rounded-xl border border-gray-200 p-6 animate-fade-in-right stagger-3 hover:border-indigo-200 hover:bg-indigo-50/30 transition-colors"
            >
              <div className="flex items-center gap-2 mb-4">
                <FileText className="h-5 w-5 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">Sample Analysis</span>
                <Badge variant="secondary" className="ml-auto">
                  Click to open
                </Badge>
              </div>
              <div className="space-y-4">
                <div className="p-4 bg-white border border-red-100 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-medium text-red-800 text-sm mb-1">
                        Mandatory Arbitration Found
                      </div>
                      <p className="text-xs text-red-700">
                        &quot;Any dispute arising out of this Agreement shall be resolved
                        by binding arbitration...&quot;
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="danger">High Risk</Badge>
                        <span className="text-xs text-red-600">94% confidence</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="p-4 bg-white border border-amber-100 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-medium text-amber-800 text-sm mb-1">
                        Class Action Waiver
                      </div>
                      <p className="text-xs text-amber-700">
                        &quot;You agree to bring claims only on an individual basis and
                        not as a class...&quot;
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="warning">Medium Risk</Badge>
                        <span className="text-xs text-amber-600">89% confidence</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-b from-white to-indigo-50/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="text-center animate-fade-in-up stagger-2">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">
              Ready to review a contract?
            </h2>
            <p className="text-gray-600 mb-8 max-w-xl mx-auto">
              Start with the sample MSA for a reliable walkthrough, then try a live
              upload if the API is available. No signup required.
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={springTransition}
              >
                <Button asChild size="lg">
                  <Link href={SAMPLE_DEMO_PATH}>
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    Open sample analysis
                  </Link>
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={springTransition}
              >
                <Button variant="outline" size="lg" asChild>
                  <Link href="/upload">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
