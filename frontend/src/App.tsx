import { useState, useCallback } from 'react';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';
import {
  FileText, Upload, Shield, ArrowLeftRight, CheckCircle, AlertTriangle,
  ChevronDown, ChevronRight, Scale, Clock, Eye,
} from 'lucide-react';
import { cn } from "@/lib/utils";

const queryClient = new QueryClient();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Detection {
  id: string;
  type: 'arbitration' | 'unfair' | 'date';
  label: string;
  section: string;
  confidence: number;
  severity: 'high' | 'medium' | 'low';
  excerpt: string;
}

interface Comparison {
  category: string;
  original: string;
  recommended: string;
  severity: 'high' | 'medium' | 'low';
}

interface ContractData {
  id: string;
  name: string;
  type: string;
  parties: string;
  date: string;
  overallRisk: number;
  riskBreakdown: { subject: string; score: number }[];
  detections: Detection[];
  comparisons: Comparison[];
  sections: { title: string; content: React.ReactNode }[];
}

// ---------------------------------------------------------------------------
// Inline Highlight
// ---------------------------------------------------------------------------

function HL({ type, confidence, children }: {
  type: 'arbitration' | 'unfair' | 'date';
  confidence: number;
  children: React.ReactNode;
}) {
  const bg = {
    arbitration: 'bg-yellow-100 border-b-2 border-yellow-400',
    unfair: 'bg-red-100 border-b-2 border-red-400',
    date: 'bg-blue-100 border-b-2 border-blue-400',
  };
  return (
    <span className={cn(bg[type], 'px-0.5 rounded-sm relative group/hl cursor-help')}>
      {children}
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover/hl:block bg-gray-900 text-white text-[11px] px-2 py-1 rounded whitespace-nowrap z-20 pointer-events-none shadow-lg">
        {confidence}% confidence
      </span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Risk Gauge (SVG circle)
// ---------------------------------------------------------------------------

function RiskGauge({ score }: { score: number }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const filled = (score / 100) * c;
  const color = score >= 70 ? '#dc2626' : score >= 40 ? '#d97706' : '#16a34a';
  const label = score >= 70 ? 'High Risk' : score >= 40 ? 'Medium Risk' : 'Low Risk';
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-40 h-40">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          <circle cx="60" cy="60" r={r} fill="none" stroke="#e5e7eb" strokeWidth="8" />
          <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
            strokeDasharray={c} strokeDashoffset={c - filled} strokeLinecap="round" />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold" style={{ color }}>{score}</span>
          <span className="text-xs text-gray-500 font-medium">/ 100</span>
        </div>
      </div>
      <span className="mt-2 text-sm font-semibold" style={{ color }}>{label}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Severity badge helper
// ---------------------------------------------------------------------------

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={cn(
      'text-[10px] font-bold uppercase px-1.5 py-0.5 rounded',
      severity === 'high' ? 'bg-red-100 text-red-700'
        : severity === 'medium' ? 'bg-yellow-100 text-yellow-700'
        : 'bg-green-100 text-green-700'
    )}>{severity}</span>
  );
}

// ---------------------------------------------------------------------------
// Contract Data
// ---------------------------------------------------------------------------

const CONTRACTS: ContractData[] = [
  {
    id: 'nda',
    name: 'Non-Disclosure Agreement',
    type: 'NDA',
    parties: 'TechVentures Inc. & Acme Corp.',
    date: 'March 15, 2026',
    overallRisk: 68,
    riskBreakdown: [
      { subject: 'Arbitration Risk', score: 82 },
      { subject: 'Liability', score: 55 },
      { subject: 'IP Assignment', score: 30 },
      { subject: 'Non-Compete', score: 15 },
    ],
    detections: [
      { id: 'nda-1', type: 'arbitration', label: 'Mandatory Binding Arbitration', section: 'Section 6', confidence: 94, severity: 'high', excerpt: 'Binding arbitration via AAA, jury trial waiver' },
      { id: 'nda-2', type: 'unfair', label: 'Injunctive Relief Without Proof', section: 'Section 5', confidence: 87, severity: 'medium', excerpt: 'No requirement to prove actual damages' },
      { id: 'nda-3', type: 'date', label: '5-Year Term + 3-Year Survival', section: 'Section 4', confidence: 99, severity: 'low', excerpt: 'Extended confidentiality obligations' },
    ],
    comparisons: [
      {
        category: 'Dispute Resolution',
        original: 'Any dispute arising out of this Agreement shall be resolved through binding arbitration in San Francisco, CA, administered by the American Arbitration Association. The arbitration shall be conducted by a single arbitrator. The parties waive any right to a jury trial.',
        recommended: 'Any dispute arising out of this Agreement shall first be submitted to mediation for a period of 30 days. If unresolved, either party may pursue resolution through binding arbitration or litigation in the courts of San Francisco, CA. Each party retains the right to seek injunctive relief in court.',
        severity: 'high',
      },
      {
        category: 'Remedies',
        original: 'The Disclosing Party shall be entitled to seek injunctive relief in any court of competent jurisdiction without the necessity of proving actual damages or posting any bond or other security.',
        recommended: 'The Disclosing Party may seek injunctive relief in any court of competent jurisdiction. The court may require the Disclosing Party to demonstrate likelihood of irreparable harm. Bond requirements shall be determined by the court.',
        severity: 'medium',
      },
    ],
    sections: [
      {
        title: 'Preamble',
        content: (
          <p className="text-gray-700 leading-relaxed">
            This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of{' '}
            <HL type="date" confidence={99}>March 15, 2026</HL>{' '}
            ("Effective Date") by and between TechVentures Inc., a Delaware corporation ("Disclosing Party"), and Acme Corp., a California corporation ("Receiving Party"), collectively referred to as the "Parties."
          </p>
        ),
      },
      {
        title: '1. Confidential Information',
        content: (
          <p className="text-gray-700 leading-relaxed">
            "Confidential Information" means any non-public information, in any form or medium, disclosed by either party to the other, including but not limited to: trade secrets, business plans, financial data, customer lists, technical specifications, software code, algorithms, and proprietary methodologies. Confidential Information includes information disclosed orally if identified as confidential within 10 business days of disclosure.
          </p>
        ),
      },
      {
        title: '2. Obligations of Receiving Party',
        content: (
          <p className="text-gray-700 leading-relaxed">
            The Receiving Party shall: (a) hold all Confidential Information in strict confidence; (b) not disclose Confidential Information to any third party without prior written consent; (c) use Confidential Information solely for the purpose of evaluating and pursuing a potential business relationship between the Parties; (d) limit access to Confidential Information to employees and advisors with a need to know who are bound by confidentiality obligations no less restrictive than those contained herein.
          </p>
        ),
      },
      {
        title: '3. Exceptions',
        content: (
          <p className="text-gray-700 leading-relaxed">
            The obligations set forth in Section 2 shall not apply to information that: (a) is or becomes publicly available through no fault of the Receiving Party; (b) was known to the Receiving Party prior to disclosure, as demonstrated by written records; (c) is independently developed by the Receiving Party without use of or reference to the Confidential Information; (d) is rightfully obtained from a third party without restriction on disclosure.
          </p>
        ),
      },
      {
        title: '4. Term and Termination',
        content: (
          <p className="text-gray-700 leading-relaxed">
            This Agreement shall remain in effect for a period of{' '}
            <HL type="date" confidence={99}>five (5) years from the Effective Date</HL>. Either party may terminate this Agreement upon 30 days' written notice. The obligations of confidentiality set forth herein shall{' '}
            <HL type="date" confidence={97}>survive for a period of three (3) years following termination</HL>{' '}
            or expiration of this Agreement.
          </p>
        ),
      },
      {
        title: '5. Remedies',
        content: (
          <p className="text-gray-700 leading-relaxed">
            The Parties acknowledge that a breach of this Agreement may cause irreparable harm for which monetary damages would be inadequate.{' '}
            <HL type="unfair" confidence={87}>
              The Disclosing Party shall be entitled to seek injunctive relief in any court of competent jurisdiction without the necessity of proving actual damages or posting any bond or other security.
            </HL>{' '}
            Such remedies shall be in addition to any other remedies available at law or in equity.
          </p>
        ),
      },
      {
        title: '6. Dispute Resolution',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="arbitration" confidence={94}>
              Any dispute, controversy, or claim arising out of or relating to this Agreement, including the breach, termination, or invalidity thereof, shall be resolved through binding arbitration in San Francisco, California, administered by the American Arbitration Association in accordance with its Commercial Arbitration Rules. The arbitration shall be conducted by a single arbitrator selected in accordance with AAA rules. The parties irrevocably waive any right to a jury trial in any action or proceeding.
            </HL>{' '}
            The arbitrator's decision shall be final and binding, and judgment upon the award may be entered in any court having jurisdiction thereof.
          </p>
        ),
      },
      {
        title: '7. Governing Law',
        content: (
          <p className="text-gray-700 leading-relaxed">
            This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of laws principles.
          </p>
        ),
      },
    ],
  },
  {
    id: 'employment',
    name: 'Employment Agreement',
    type: 'Employment',
    parties: 'GlobalTech Solutions & Employee',
    date: 'January 1, 2026',
    overallRisk: 74,
    riskBreakdown: [
      { subject: 'Arbitration Risk', score: 78 },
      { subject: 'Liability', score: 42 },
      { subject: 'IP Assignment', score: 85 },
      { subject: 'Non-Compete', score: 90 },
    ],
    detections: [
      { id: 'emp-1', type: 'arbitration', label: 'Binding Arbitration + Class Waiver', section: 'Section 5', confidence: 92, severity: 'high', excerpt: 'JAMS arbitration with class action waiver' },
      { id: 'emp-2', type: 'unfair', label: 'Excessive Non-Compete Scope', section: 'Section 3', confidence: 89, severity: 'high', excerpt: '24 months, 150-mile radius restriction' },
      { id: 'emp-3', type: 'unfair', label: 'Broad IP Assignment', section: 'Section 4', confidence: 85, severity: 'medium', excerpt: 'Includes work outside business hours' },
      { id: 'emp-4', type: 'date', label: 'Asymmetric Notice Periods', section: 'Section 6', confidence: 97, severity: 'low', excerpt: '30 days employer vs 90 days employee' },
    ],
    comparisons: [
      {
        category: 'Non-Compete Clause',
        original: 'Employee agrees not to engage in any business that competes with Employer for a period of twenty-four (24) months following termination, within a 150-mile radius of any Employer office.',
        recommended: 'Employee agrees not to engage in directly competitive business activities for a period of twelve (12) months following termination, limited to the Employee\'s specific area of expertise and within a 25-mile radius of Employee\'s primary work location.',
        severity: 'high',
      },
      {
        category: 'Intellectual Property',
        original: 'All work product created during employment, including work created outside of business hours using personal equipment, shall be the exclusive property of Employer.',
        recommended: 'All work product created during business hours or using Employer resources, and directly related to Employee\'s role, shall be the property of Employer. Work created on personal time using personal equipment and unrelated to Employer\'s business remains Employee\'s property.',
        severity: 'medium',
      },
    ],
    sections: [
      {
        title: 'Preamble',
        content: (
          <p className="text-gray-700 leading-relaxed">
            This Employment Agreement ("Agreement") is made effective as of{' '}
            <HL type="date" confidence={99}>January 1, 2026</HL>{' '}
            by and between GlobalTech Solutions, a New York corporation ("Employer"), and the undersigned individual ("Employee").
          </p>
        ),
      },
      {
        title: '1. Position and Duties',
        content: (
          <p className="text-gray-700 leading-relaxed">
            Employee is hired for the position of Senior Software Engineer. Employee shall devote full business time and best efforts to performing duties as assigned by Employer. Employee shall report to the VP of Engineering and shall comply with all company policies and procedures.
          </p>
        ),
      },
      {
        title: '2. Compensation',
        content: (
          <p className="text-gray-700 leading-relaxed">
            Base salary of $185,000 per year, paid bi-weekly. Employee shall be eligible for an annual performance bonus of up to 20% of base salary, at Employer's sole discretion.{' '}
            <HL type="date" confidence={96}>First performance review shall occur on July 1, 2026.</HL>{' '}
            Benefits include health insurance, 401(k) matching up to 4%, and 20 days of paid time off per year.
          </p>
        ),
      },
      {
        title: '3. Non-Compete',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="unfair" confidence={89}>
              Employee agrees not to engage in any business that competes directly or indirectly with Employer for a period of twenty-four (24) months following termination of employment for any reason, within a 150-mile radius of any Employer office location.
            </HL>{' '}
            This restriction applies to employment, consulting, ownership, or any other business relationship with a competing entity.
          </p>
        ),
      },
      {
        title: '4. Intellectual Property',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="unfair" confidence={85}>
              All inventions, discoveries, improvements, and work product conceived or created by Employee during the term of employment, including work created outside of business hours using personal equipment, shall be the sole and exclusive property of Employer.
            </HL>{' '}
            Employee hereby assigns all rights, title, and interest in such work product to Employer. Employee agrees to execute any documents necessary to perfect Employer's ownership rights.
          </p>
        ),
      },
      {
        title: '5. Dispute Resolution',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="arbitration" confidence={92}>
              Any dispute or claim arising out of or relating to this Agreement, Employee's employment, or the termination thereof, shall be settled by final and binding arbitration under the rules of JAMS in New York, NY. The arbitration shall be conducted by a single arbitrator. Employee waives the right to participate in any class, collective, or representative action or arbitration.
            </HL>{' '}
            Each party shall bear their own costs and attorney's fees unless the arbitrator determines otherwise.
          </p>
        ),
      },
      {
        title: '6. Termination',
        content: (
          <p className="text-gray-700 leading-relaxed">
            This Agreement may be terminated: (a) by Employer for cause with{' '}
            <HL type="date" confidence={97}>30 days' written notice</HL>; (b) by Employee with{' '}
            <HL type="date" confidence={97}>90 days' written notice</HL>. In the event of termination without cause, Employee shall receive severance equal to 4 weeks' salary per year of service, up to a maximum of 26 weeks.
          </p>
        ),
      },
    ],
  },
  {
    id: 'saas',
    name: 'SaaS Terms of Service',
    type: 'SaaS ToS',
    parties: 'CloudPlatform Pro & Customer',
    date: 'February 1, 2026',
    overallRisk: 58,
    riskBreakdown: [
      { subject: 'Arbitration Risk', score: 75 },
      { subject: 'Liability', score: 68 },
      { subject: 'IP Assignment', score: 20 },
      { subject: 'Non-Compete', score: 10 },
    ],
    detections: [
      { id: 'saas-1', type: 'arbitration', label: 'Individual Binding Arbitration', section: 'Section 4', confidence: 93, severity: 'high', excerpt: 'AAA arbitration with class action waiver' },
      { id: 'saas-2', type: 'unfair', label: 'Broad Liability Limitation', section: 'Section 3', confidence: 86, severity: 'medium', excerpt: 'No consequential damages, 12-month fee cap' },
      { id: 'saas-3', type: 'date', label: 'Auto-Renewal with Price Increase', section: 'Section 2', confidence: 95, severity: 'low', excerpt: '30-day cancellation window, up to 10% increase' },
    ],
    comparisons: [
      {
        category: 'Dispute Resolution',
        original: 'All disputes shall be resolved through individual binding arbitration in accordance with the AAA Consumer Arbitration Rules. You waive your right to participate in class action lawsuits or class-wide arbitration.',
        recommended: 'Disputes under $10,000 may be resolved in small claims court. For larger disputes, parties agree to attempt mediation for 30 days before proceeding to binding arbitration under AAA Consumer Arbitration Rules. Class action rights are preserved for claims involving systemic issues.',
        severity: 'high',
      },
      {
        category: 'Liability Limitation',
        original: 'Provider\'s total liability shall not exceed the fees paid in the twelve (12) months preceding the claim. Provider shall not be liable for any indirect, incidental, or consequential damages, including lost profits.',
        recommended: 'Provider\'s total liability shall not exceed the greater of (a) fees paid in the twelve (12) months preceding the claim or (b) $50,000. The limitation on consequential damages shall not apply to breaches of data security obligations or willful misconduct.',
        severity: 'medium',
      },
    ],
    sections: [
      {
        title: 'Preamble',
        content: (
          <p className="text-gray-700 leading-relaxed">
            These Terms of Service ("Terms") govern your access to and use of CloudPlatform Pro, effective as of{' '}
            <HL type="date" confidence={99}>February 1, 2026</HL>. By accessing or using the Service, you agree to be bound by these Terms. "Customer" or "you" refers to the entity or individual agreeing to these Terms.
          </p>
        ),
      },
      {
        title: '1. Service and SLA',
        content: (
          <p className="text-gray-700 leading-relaxed">
            CloudPlatform Pro provides cloud-based project management and collaboration tools on a subscription basis. Provider targets 99.9% monthly uptime for all paid plans. Service credits of 5% of monthly fees will be issued for each 0.1% below the target SLA, up to a maximum credit of 30% of the applicable monthly fee. Scheduled maintenance windows are excluded from uptime calculations.
          </p>
        ),
      },
      {
        title: '2. Subscription and Payment',
        content: (
          <p className="text-gray-700 leading-relaxed">
            Monthly subscription: $49 per user per month. Annual subscription: $39 per user per month (billed annually). All fees are non-refundable except as expressly stated in these Terms.{' '}
            <HL type="date" confidence={95}>
              Subscriptions automatically renew for successive terms of equal length unless cancelled at least 30 days before the renewal date. Provider may increase prices by up to 10% upon renewal with at least 15 days' prior notice.
            </HL>
          </p>
        ),
      },
      {
        title: '3. Liability Limitation',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="unfair" confidence={86}>
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, PROVIDER'S TOTAL AGGREGATE LIABILITY FOR ALL CLAIMS ARISING OUT OF OR RELATING TO THESE TERMS SHALL NOT EXCEED THE TOTAL FEES PAID BY CUSTOMER IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM. IN NO EVENT SHALL PROVIDER BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT LIMITED TO LOSS OF PROFITS, DATA, OR BUSINESS OPPORTUNITIES, REGARDLESS OF THE THEORY OF LIABILITY.
            </HL>
          </p>
        ),
      },
      {
        title: '4. Dispute Resolution',
        content: (
          <p className="text-gray-700 leading-relaxed">
            <HL type="arbitration" confidence={93}>
              Any dispute, claim, or controversy arising out of or relating to these Terms or the Service, including the determination of the scope or applicability of this agreement to arbitrate, shall be resolved through individual binding arbitration in accordance with the American Arbitration Association's Consumer Arbitration Rules. By agreeing to these Terms, you waive your right to participate in class action lawsuits, class-wide arbitration, or any consolidated or representative proceedings.
            </HL>{' '}
            Notwithstanding the foregoing, either party may seek injunctive relief in any court of competent jurisdiction. Small claims court actions are excluded from this arbitration provision.
          </p>
        ),
      },
      {
        title: '5. Termination',
        content: (
          <p className="text-gray-700 leading-relaxed">
            Provider may suspend or terminate Service for violation of these Terms with{' '}
            <HL type="date" confidence={92}>5 business days' written notice</HL>.
            Customer may terminate at any time by cancelling the subscription through the account settings. No refunds will be issued for partial billing periods. Upon termination, Customer's data will be retained for 30 days before permanent deletion.
          </p>
        ),
      },
    ],
  },
];

// ---------------------------------------------------------------------------
// Tab definitions
// ---------------------------------------------------------------------------

const TABS = [
  { id: 'documents', label: 'Document Analysis', icon: FileText },
  { id: 'upload', label: 'Upload', icon: Upload },
  { id: 'risk', label: 'Risk Dashboard', icon: Shield },
  { id: 'compare', label: 'Clause Comparison', icon: ArrowLeftRight },
] as const;

// ---------------------------------------------------------------------------
// Document Analysis Tab
// ---------------------------------------------------------------------------

function DocumentAnalysis({ contracts, selectedId, onSelect }: {
  contracts: ContractData[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const selected = contracts.find(c => c.id === selectedId) || contracts[0];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Sidebar: contract list */}
      <div className="lg:col-span-1">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Sample Contracts</h2>
        <div className="space-y-2">
          {contracts.map(c => (
            <button key={c.id} onClick={() => onSelect(c.id)}
              className={cn(
                'w-full text-left p-4 rounded-lg border transition-all',
                selectedId === c.id
                  ? 'bg-white border-blue-700 shadow-sm ring-1 ring-blue-700/20'
                  : 'bg-white border-gray-200 hover:border-gray-300 hover:shadow-sm'
              )}
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-900">{c.name}</p>
                  <p className="text-xs text-gray-500 mt-0.5">{c.type}</p>
                </div>
                <span className={cn(
                  'text-xs font-semibold px-2 py-0.5 rounded-full',
                  c.overallRisk >= 70 ? 'bg-red-100 text-red-700'
                    : c.overallRisk >= 40 ? 'bg-yellow-100 text-yellow-700'
                    : 'bg-green-100 text-green-700'
                )}>{c.overallRisk}</span>
              </div>
              <div className="flex gap-3 mt-2 text-xs text-gray-400">
                <span>{c.detections.length} issues</span>
                <span>{c.date}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main: document content */}
      <div className="lg:col-span-3 space-y-5">
        {/* Header card */}
        <div className="bg-white rounded-lg border border-gray-200 p-5">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">{selected.name}</h2>
              <p className="text-sm text-gray-500 mt-1">{selected.parties} &middot; {selected.date}</p>
            </div>
            <div className="text-right">
              <div className="text-xs text-gray-500">Risk Score</div>
              <div className={cn(
                'text-2xl font-bold',
                selected.overallRisk >= 70 ? 'text-red-600' : selected.overallRisk >= 40 ? 'text-yellow-600' : 'text-green-600'
              )}>{selected.overallRisk}/100</div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {selected.detections.map(d => (
              <span key={d.id} className={cn(
                'inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border',
                d.type === 'arbitration' ? 'bg-yellow-50 border-yellow-200 text-yellow-800'
                  : d.type === 'unfair' ? 'bg-red-50 border-red-200 text-red-800'
                  : 'bg-blue-50 border-blue-200 text-blue-800'
              )}>
                {d.type === 'arbitration' && <AlertTriangle className="w-3 h-3" />}
                {d.type === 'unfair' && <Shield className="w-3 h-3" />}
                {d.type === 'date' && <Clock className="w-3 h-3" />}
                {d.label} &middot; {d.confidence}%
              </span>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-6 text-xs text-gray-600 px-1">
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-2 bg-yellow-200 border border-yellow-400 rounded-sm" /> Arbitration Clause
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-2 bg-red-200 border border-red-400 rounded-sm" /> Unfair Term
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-4 h-2 bg-blue-200 border border-blue-400 rounded-sm" /> Key Date
          </span>
          <span className="text-gray-400 ml-auto">Hover highlights for confidence scores</span>
        </div>

        {/* Document sections */}
        <div className="bg-white rounded-lg border border-gray-200 divide-y divide-gray-100">
          {selected.sections.map((section, i) => (
            <div key={i} className="p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">{section.title}</h3>
              {section.content}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Upload Flow Tab
// ---------------------------------------------------------------------------

function UploadFlow() {
  const [isDragging, setIsDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [step, setStep] = useState('');
  const [progress, setProgress] = useState(0);
  const [complete, setComplete] = useState(false);

  const startAnalysis = useCallback(() => {
    setAnalyzing(true);
    setComplete(false);
    setProgress(0);
    const steps = [
      'Parsing document structure...',
      'Extracting text content...',
      'Running clause detection model...',
      'Computing risk scores...',
      'Generating recommendations...',
    ];
    let i = 0;
    const interval = setInterval(() => {
      if (i < steps.length) {
        setStep(steps[i]);
        setProgress(((i + 1) / steps.length) * 100);
        i++;
      } else {
        clearInterval(interval);
        setAnalyzing(false);
        setComplete(true);
      }
    }, 900);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    startAnalysis();
  }, [startAnalysis]);

  const DEMO_RESULTS = [
    { severity: 'high', label: 'Mandatory Binding Arbitration', section: 'Section 8.1', confidence: 94 },
    { severity: 'high', label: 'Class Action Waiver', section: 'Section 8.3', confidence: 91 },
    { severity: 'medium', label: 'Liability Cap Below Market', section: 'Section 6.2', confidence: 88 },
    { severity: 'low', label: 'Auto-Renewal Terms', section: 'Section 3.4', confidence: 96 },
  ];

  if (complete) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Analysis Complete</h2>
              <p className="text-sm text-gray-500">Contract_Draft_2026.pdf</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">72</div>
              <div className="text-xs text-gray-500 mt-1">Risk Score</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">4</div>
              <div className="text-xs text-gray-500 mt-1">Issues Found</div>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-700">4.5s</div>
              <div className="text-xs text-gray-500 mt-1">Processing</div>
            </div>
          </div>
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Detected Issues</h3>
          <div className="space-y-2 mb-6">
            {DEMO_RESULTS.map((d, i) => (
              <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-gray-50">
                <div className="flex items-center gap-2">
                  <SeverityBadge severity={d.severity} />
                  <span className="text-sm text-gray-900">{d.label}</span>
                </div>
                <span className="text-xs text-gray-400">{d.section} &middot; {d.confidence}%</span>
              </div>
            ))}
          </div>
          <button onClick={() => { setComplete(false); }}
            className="w-full py-2.5 bg-blue-700 text-white text-sm font-medium rounded-lg hover:bg-blue-800 transition-colors">
            Upload Another Document
          </button>
        </div>
      </div>
    );
  }

  if (analyzing) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg border border-gray-200 p-8">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Analyzing Document</h2>
          <p className="text-sm text-gray-500 mb-6">Contract_Draft_2026.pdf</p>
          <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
            <div className="bg-blue-700 h-2 rounded-full transition-all duration-700"
              style={{ width: `${progress}%` }} />
          </div>
          <p className="text-sm text-gray-600">{step}</p>
          <p className="text-xs text-gray-400 mt-1">{Math.round(progress)}% complete</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          'border-2 border-dashed rounded-lg p-16 text-center transition-all',
          isDragging ? 'border-blue-700 bg-blue-50' : 'border-gray-300 bg-white hover:border-gray-400'
        )}
      >
        <Upload className={cn('w-12 h-12 mx-auto mb-4', isDragging ? 'text-blue-700' : 'text-gray-400')} />
        <p className="text-lg font-medium text-gray-700 mb-1">
          {isDragging ? 'Drop your file here' : 'Drag and drop your contract'}
        </p>
        <p className="text-sm text-gray-500 mb-6">PDF, DOCX, or TXT files up to 50MB</p>
        <button onClick={startAnalysis}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-700 text-white text-sm font-medium rounded-lg hover:bg-blue-800 transition-colors">
          <Upload className="w-4 h-4" />
          Choose File or Run Demo
        </button>
        <p className="text-xs text-gray-400 mt-4">Click the button to run a demo analysis</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Risk Dashboard Tab
// ---------------------------------------------------------------------------

function RiskDashboard({ contract }: { contract: ContractData }) {
  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-500">
        Risk analysis for: <span className="font-medium text-gray-900">{contract.name}</span>
        <span className="text-gray-400 ml-1">(switch contracts in Document Analysis tab)</span>
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Overall Score */}
        <div className="bg-white rounded-lg border border-gray-200 p-6 flex flex-col items-center justify-center">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-6">Overall Contract Risk</h3>
          <RiskGauge score={contract.overallRisk} />
        </div>

        {/* Radar Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">Risk Breakdown</h3>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={contract.riskBreakdown}>
              <PolarGrid stroke="#e5e7eb" />
              <PolarAngleAxis dataKey="subject" tick={{ fill: '#6b7280', fontSize: 11 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <Radar name="Risk" dataKey="score" stroke="#1d4ed8" fill="#1d4ed8" fillOpacity={0.15} strokeWidth={2} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Category cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {contract.riskBreakdown.map(item => {
          const color = item.score >= 70 ? 'red' : item.score >= 40 ? 'yellow' : 'green';
          return (
            <div key={item.subject} className="bg-white rounded-lg border border-gray-200 p-4">
              <h4 className="text-xs font-medium text-gray-500 mb-2">{item.subject}</h4>
              <div className="flex items-end justify-between mb-2">
                <span className={cn(
                  'text-3xl font-bold',
                  color === 'red' ? 'text-red-600' : color === 'yellow' ? 'text-yellow-600' : 'text-green-600'
                )}>{item.score}</span>
                <span className="text-xs text-gray-400">/ 100</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5">
                <div className={cn(
                  'h-1.5 rounded-full transition-all',
                  color === 'red' ? 'bg-red-500' : color === 'yellow' ? 'bg-yellow-500' : 'bg-green-500'
                )} style={{ width: `${item.score}%` }} />
              </div>
            </div>
          );
        })}
      </div>

      {/* Detected issues */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="p-4 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-900">Detected Issues ({contract.detections.length})</h3>
        </div>
        <div className="divide-y divide-gray-100">
          {contract.detections.map(d => (
            <div key={d.id} className="p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <SeverityBadge severity={d.severity} />
                <div>
                  <p className="text-sm font-medium text-gray-900">{d.label}</p>
                  <p className="text-xs text-gray-500">{d.section} &middot; {d.excerpt}</p>
                </div>
              </div>
              <span className="text-sm font-medium text-gray-600">{d.confidence}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Clause Comparison Tab
// ---------------------------------------------------------------------------

function ClauseComparison({ contract }: { contract: ContractData }) {
  return (
    <div className="space-y-6">
      <p className="text-sm text-gray-500">
        Clause revisions for: <span className="font-medium text-gray-900">{contract.name}</span>
      </p>

      {contract.comparisons.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <CheckCircle className="w-10 h-10 text-green-500 mx-auto mb-3" />
          <p className="text-gray-600 font-medium">No problematic clauses found in this contract.</p>
        </div>
      ) : (
        contract.comparisons.map((comp, i) => (
          <div key={i} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="p-4 border-b border-gray-100 flex items-center justify-between bg-gray-50">
              <h3 className="text-sm font-semibold text-gray-900">{comp.category}</h3>
              <SeverityBadge severity={comp.severity} />
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-gray-100">
              <div className="p-5">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-2 h-2 bg-red-500 rounded-full" />
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Original Clause</h4>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-red-50 border border-red-200 rounded-lg p-4">{comp.original}</p>
              </div>
              <div className="p-5">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-2 h-2 bg-green-500 rounded-full" />
                  <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Recommended Revision</h4>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed bg-green-50 border border-green-200 rounded-lg p-4">{comp.recommended}</p>
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

function MainApp() {
  const [activeTab, setActiveTab] = useState('documents');
  const [selectedContractId, setSelectedContractId] = useState('nda');
  const selectedContract = CONTRACTS.find(c => c.id === selectedContractId) || CONTRACTS[0];

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-blue-700 rounded-lg flex items-center justify-center">
              <Scale className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gray-900 tracking-tight">Smart Legal Contracts</h1>
              <p className="text-[11px] text-gray-500 font-medium tracking-wide">AI-Powered Analysis</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden sm:flex items-center gap-2 text-xs text-gray-500 bg-gray-100 px-3 py-1.5 rounded-full">
              <span className="w-1.5 h-1.5 bg-green-500 rounded-full" />
              System Online
            </span>
          </div>
        </div>
        {/* Tabs */}
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-0 -mb-px">
            {TABS.map(tab => {
              const Icon = tab.icon;
              return (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    'flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors',
                    activeTab === tab.id
                      ? 'border-blue-700 text-blue-700'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  )}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'documents' && (
          <DocumentAnalysis
            contracts={CONTRACTS}
            selectedId={selectedContractId}
            onSelect={setSelectedContractId}
          />
        )}
        {activeTab === 'upload' && <UploadFlow />}
        {activeTab === 'risk' && <RiskDashboard contract={selectedContract} />}
        {activeTab === 'compare' && <ClauseComparison contract={selectedContract} />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-gray-400">
          <span>Smart Legal Contracts &copy; 2026</span>
          <span>FastAPI + ChromaDB + React + BERT NLP</span>
        </div>
      </footer>
    </div>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <MainApp />
  </QueryClientProvider>
);

export default App;
