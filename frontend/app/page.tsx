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
} from "lucide-react";
import { CountUp } from "@/components/ui/motion";

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
    description:
      "Compare clauses across documents to identify differences and similarities.",
  },
  {
    icon: Clock,
    title: "Analysis History",
    description:
      "Track all your analyses with filtering, search, and export capabilities.",
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
      {/* Hero Section */}
      <section
        ref={heroRef}
        className="relative border-b border-gray-100 bg-white overflow-hidden"
      >
        <motion.div
          className="absolute inset-0 bg-gradient-to-br from-blue-50/50 via-white to-indigo-50/30"
          style={{ y: bgY }}
        />
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-24">
          <div className="max-w-3xl">
            <Badge variant="secondary" className="mb-4 animate-fade-in-up">
              AI-Powered Legal Analysis
            </Badge>
            <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight mb-4 gradient-text animate-fade-in-up stagger-1">
              Detect arbitration clauses in seconds
            </h1>
            <p className="text-lg text-gray-600 mb-8 leading-relaxed animate-fade-in-up stagger-2">
              Upload your legal documents and instantly identify arbitration
              clauses, jury waivers, and dispute resolution provisions. Get
              detailed risk assessments and actionable insights.
            </p>
            <div className="flex flex-wrap gap-3 animate-fade-in-up stagger-3">
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={springTransition}
              >
                <Button asChild size="lg">
                  <Link href="/upload">
                    <Upload className="mr-2 h-4 w-4" />
                    Upload Document
                  </Link>
                </Button>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={springTransition}
              >
                <Button variant="outline" size="lg" asChild>
                  <Link href="/history">
                    View Demo
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </motion.div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
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
                    <CountUp
                      end={stat.value}
                      prefix={stat.prefix}
                      suffix={stat.suffix}
                    />
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

      {/* Features Grid */}
      <section className="border-b border-gray-100 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="text-center mb-12 animate-fade-in-up">
            <h2 className="text-2xl font-semibold text-gray-900 mb-3">
              Everything you need to analyze legal documents
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our platform combines advanced NLP with retrieval-augmented
              generation to deliver accurate, fast, and actionable insights.
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
                          <h3 className="font-medium text-gray-900 mb-1">
                            {feature.title}
                          </h3>
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

      {/* Clause Types Section */}
      <section className="border-b border-gray-100 bg-gray-50/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="animate-fade-in-left stagger-2">
              <h2 className="text-2xl font-semibold text-gray-900 mb-4">
                Comprehensive clause detection
              </h2>
              <p className="text-gray-600 mb-6 leading-relaxed">
                Our AI model is trained to identify various types of arbitration
                and dispute resolution clauses commonly found in contracts and
                legal agreements.
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
            <div className="bg-white rounded-lg border border-gray-200 p-6 animate-fade-in-right stagger-3">
              <div className="flex items-center gap-2 mb-4">
                <FileText className="h-5 w-5 text-gray-400" />
                <span className="text-sm font-medium text-gray-700">
                  Sample Analysis
                </span>
              </div>
              <div className="space-y-4">
                <div className="p-4 bg-red-50 border border-red-100 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-medium text-red-800 text-sm mb-1">
                        Mandatory Arbitration Found
                      </div>
                      <p className="text-xs text-red-700">
                        &quot;Any dispute arising out of this Agreement shall be
                        resolved by binding arbitration...&quot;
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="danger">High Risk</Badge>
                        <span className="text-xs text-red-600">
                          94% confidence
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="p-4 bg-amber-50 border border-amber-100 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <div className="font-medium text-amber-800 text-sm mb-1">
                        Class Action Waiver
                      </div>
                      <p className="text-xs text-amber-700">
                        &quot;You agree to bring claims only on an individual
                        basis and not as a class...&quot;
                      </p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant="warning">Medium Risk</Badge>
                        <span className="text-xs text-amber-600">
                          89% confidence
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
          <div className="text-center animate-fade-in-up stagger-2">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">
              Ready to analyze your documents?
            </h2>
            <p className="text-gray-600 mb-8 max-w-xl mx-auto">
              Upload your first document and see our AI-powered analysis in
              action. No signup required for basic analysis.
            </p>
            <div className="flex flex-wrap justify-center gap-3">
              <motion.div
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={springTransition}
              >
                <Button asChild size="lg">
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
