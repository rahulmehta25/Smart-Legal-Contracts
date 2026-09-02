"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Reveal } from "@/components/ui/reveal";
import { TextLink } from "@/components/ui/text-link";
import { CountUp } from "@/components/ui/motion";
import { SAMPLE_DEMO_PATH, SAMPLE_DOCUMENT } from "@/lib/sample-analysis";

const stats = [
  { value: 85, label: "Detection accuracy", prefix: "", suffix: "%+" },
  { value: 2, label: "Typical analysis time", prefix: "<", suffix: "s" },
  { value: 50, label: "Maximum file size", prefix: "", suffix: "MB" },
  { value: null, label: "Supported formats", display: "PDF / DOCX" },
];

const steps = [
  {
    step: "01",
    title: "Bring a contract",
    description:
      "Drop a PDF or DOCX, or skip the live API and open the sample SaaS MSA.",
  },
  {
    step: "02",
    title: "Read the dispute language",
    description:
      "The model flags arbitration, jury waivers, class waivers, and forum terms.",
  },
  {
    step: "03",
    title: "Decide what to negotiate",
    description:
      "Each finding includes the quoted clause, a risk reading, and next steps.",
  },
];

const features = [
  {
    title: "Clause detection",
    description: "Arbitration, jury waivers, and dispute provisions, quoted in place.",
  },
  {
    title: "Risk reading",
    description: "High, medium, or low, with impact written in plain language.",
  },
  {
    title: "Fast review",
    description: "Documents typically return in seconds, with 85%+ detection accuracy.",
  },
  {
    title: "Batch processing",
    description: "Queue several files and watch progress as each analysis completes.",
  },
  {
    title: "Document comparison",
    description: "Place two analyses side by side to see which terms diverge.",
  },
  {
    title: "History",
    description: "Return to prior reviews, including the always-available sample.",
  },
];

const clauseTypes = [
  { name: "Mandatory arbitration", risk: "High" },
  { name: "Jury trial waiver", risk: "High" },
  { name: "Class action waiver", risk: "Medium" },
  { name: "Forum selection", risk: "Medium" },
  { name: "Mediation first", risk: "Low" },
  { name: "Escalation clauses", risk: "Low" },
];

export default function HomePage() {
  return (
    <div className="flex flex-col">
      <section className="band band-ivory">
        <div className="page-wrap grid items-start gap-12 py-20 lg:grid-cols-12 lg:gap-10 lg:py-28">
          <div className="lg:col-span-6">
            <p className="eyebrow animate-fade-rise">Contract review</p>
            <h1 className="display mt-5 max-w-3xl text-[2.6rem] sm:text-5xl lg:text-[3.6rem] animate-fade-rise stagger-1">
              Know the dispute terms before you sign.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-muted animate-fade-rise stagger-2">
              Smart Legal Contracts finds arbitration clauses, jury waivers, and class-action
              bars. Open the sample review if live upload is unavailable.
            </p>
            <div className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-4 animate-fade-rise stagger-3">
              <Button asChild size="lg">
                <Link href={SAMPLE_DEMO_PATH}>View Demo</Link>
              </Button>
              <TextLink href="/upload">Upload a document</TextLink>
            </div>
            <p className="mt-5 text-sm text-ink-muted animate-fade-rise stagger-4">
              Sample file: {SAMPLE_DOCUMENT.filename}
            </p>
          </div>

          <div className="lg:col-span-6 animate-fade-rise stagger-2">
            <Link
              href={SAMPLE_DEMO_PATH}
              className="group block h-full"
              aria-label="Open sample analysis"
            >
              <article className="paper-sheet relative h-full px-6 py-7 sm:px-8">
                <div className="absolute bottom-0 left-0 top-0 w-[2px] bg-brass" />
                <p className="eyebrow">Sample analysis</p>
                <h2 className="filename-display mt-3 font-serif text-2xl font-medium tracking-tight text-ink">
                  {SAMPLE_DOCUMENT.filename}
                </h2>
                <p className="mt-1 text-sm text-ink-muted">High overall risk · 92% confidence</p>

                <blockquote className="mt-6 border-l border-rule pl-4">
                  <p className="font-serif text-[1.05rem] leading-snug text-ink">
                    Mandatory arbitration
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-ink-muted">
                    Binding AAA arbitration for disputes arising out of this Agreement.
                  </p>
                  <p className="mt-2 text-xs uppercase tracking-[0.14em] text-ink-muted">
                    High risk · 94%
                  </p>
                </blockquote>

                <blockquote className="mt-5 border-l border-rule pl-4">
                  <p className="font-serif text-[1.05rem] leading-snug text-ink">
                    Class action waiver
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-ink-muted">
                    Claims may be brought only on an individual basis.
                  </p>
                  <p className="mt-2 text-xs uppercase tracking-[0.14em] text-ink-muted">
                    Medium risk · 89%
                  </p>
                </blockquote>

                <blockquote className="mt-5 border-l border-rule pl-4">
                  <p className="font-serif text-[1.05rem] leading-snug text-ink">
                    Jury trial waiver
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-ink-muted">
                    Each party waives any right to a jury trial in related proceedings.
                  </p>
                  <p className="mt-2 text-xs uppercase tracking-[0.14em] text-ink-muted">
                    High risk · 91%
                  </p>
                </blockquote>

                <dl className="mt-7 grid grid-cols-3 gap-4 border-t border-rule pt-5 text-sm">
                  <div>
                    <dt className="eyebrow">Clauses</dt>
                    <dd className="mt-1 font-serif text-xl text-ink">5</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">Risk</dt>
                    <dd className="mt-1 font-serif text-xl text-ink">High</dd>
                  </div>
                  <div>
                    <dt className="eyebrow">Confidence</dt>
                    <dd className="mt-1 font-serif text-xl text-ink">92%</dd>
                  </div>
                </dl>

                <p className="text-link mt-6 text-sm">Open the full sample</p>
              </article>
            </Link>
          </div>
        </div>
      </section>

      <div className="flex justify-center band-ivory" aria-hidden>
        <div className="h-10 w-px bg-rule" />
      </div>

      <section className="band band-linen">
        <div className="page-wrap py-14 lg:py-16">
          <Reveal>
            <div className="grid grid-cols-2 gap-10 md:grid-cols-4 md:gap-0">
              {stats.map((stat, i) => (
                <div
                  key={stat.label}
                  className={
                    i > 0
                      ? "md:border-l md:border-rule md:px-8"
                      : "md:pr-8"
                  }
                >
                  <div className="font-serif text-3xl font-medium tracking-tight text-ink md:text-4xl">
                    {stat.value !== null ? (
                      <CountUp end={stat.value} prefix={stat.prefix} suffix={stat.suffix} />
                    ) : (
                      stat.display
                    )}
                  </div>
                  <p className="mt-2 text-sm text-ink-muted">{stat.label}</p>
                </div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      <div className="flex justify-center band-linen" aria-hidden>
        <div className="h-10 w-px bg-rule" />
      </div>

      <section className="band band-ivory">
        <div className="page-wrap grid gap-16 py-20 lg:grid-cols-12 lg:py-24">
          <div className="lg:col-span-5">
            <Reveal>
              <p className="eyebrow">How it works</p>
              <h2 className="display mt-4 text-3xl sm:text-4xl">
                Three steps from document to a readable risk memo.
              </h2>
              <p className="mt-5 max-w-md text-base leading-relaxed text-ink-muted">
                Prefer a walkthrough that does not depend on the upload API? Use View Demo.
              </p>
            </Reveal>
          </div>
          <div className="lg:col-span-7">
            <ol>
              {steps.map((item, i) => (
                <li key={item.step} className="scroll-connector relative pl-12 pb-12 last:pb-0">
                  <span className="absolute left-0 top-0 font-serif text-lg text-ink-muted">
                    {item.step}
                  </span>
                  <Reveal delay={i * 0.08}>
                    <h3 className="font-serif text-2xl font-medium tracking-tight text-ink">
                      {item.title}
                    </h3>
                    <p className="mt-2 max-w-md text-base leading-relaxed text-ink-muted">
                      {item.description}
                    </p>
                  </Reveal>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </section>

      <div className="flex justify-center band-ivory" aria-hidden>
        <div className="h-10 w-px bg-rule" />
      </div>

      <section className="band band-linen">
        <div className="page-wrap py-20 lg:py-24">
          <Reveal>
            <p className="eyebrow">Capabilities</p>
            <h2 className="display mt-4 max-w-2xl text-3xl sm:text-4xl">
              The tools a review actually needs.
            </h2>
          </Reveal>
          <div className="mt-12 grid gap-x-16 md:grid-cols-2">
            {features.map((feature, i) => (
              <Reveal key={feature.title} delay={(i % 2) * 0.06}>
                <div className="border-t border-rule py-7">
                  <h3 className="font-serif text-xl font-medium tracking-tight text-ink">
                    {feature.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-ink-muted">
                    {feature.description}
                  </p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <div className="flex justify-center band-linen" aria-hidden>
        <div className="h-10 w-px bg-rule" />
      </div>

      <section className="band band-ivory">
        <div className="page-wrap grid items-start gap-16 py-20 lg:grid-cols-12 lg:py-24">
          <div className="lg:col-span-5">
            <Reveal>
              <p className="eyebrow">Clause index</p>
              <h2 className="display mt-4 text-3xl sm:text-4xl">
                The language that closes the courthouse door.
              </h2>
              <p className="mt-5 text-base leading-relaxed text-ink-muted">
                The model is trained on arbitration and dispute resolution clauses common in
                commercial agreements.
              </p>
            </Reveal>
          </div>
          <div className="lg:col-span-7">
            <Reveal>
              <ul>
                {clauseTypes.map((clause) => (
                  <li
                    key={clause.name}
                    className="flex items-baseline justify-between gap-6 border-t border-rule py-4 last:border-b"
                  >
                    <span className="font-serif text-lg text-ink">{clause.name}</span>
                    <span className="text-xs uppercase tracking-[0.16em] text-ink-muted">
                      {clause.risk} risk
                    </span>
                  </li>
                ))}
              </ul>
              <p className="mt-8">
                <TextLink href={SAMPLE_DEMO_PATH}>See these findings in the sample MSA</TextLink>
              </p>
            </Reveal>
          </div>
        </div>
      </section>

      <section className="band band-ink">
        <div className="page-wrap py-20 lg:py-24">
          <Reveal>
            <p className="text-[0.7rem] font-medium uppercase tracking-[0.22em] text-ivory/55">
              Ready when you are
            </p>
            <h2 className="mt-4 max-w-2xl font-serif text-3xl font-medium tracking-tight text-ivory sm:text-4xl">
              The sample is ready when upload is not.
            </h2>
            <p className="mt-5 max-w-xl text-base leading-relaxed text-ivory/70">
              Start with the canned MSA for a reliable walkthrough, then try a live file if
              the API is available. No signup required.
            </p>
            <div className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-4">
              <Button asChild size="lg">
                <Link href={SAMPLE_DEMO_PATH}>Open sample analysis</Link>
              </Button>
              <Link href="/upload" className="text-link text-ivory">
                Upload a document
              </Link>
            </div>
          </Reveal>
        </div>
      </section>
    </div>
  );
}
