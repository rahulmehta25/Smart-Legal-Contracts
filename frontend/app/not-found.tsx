import Link from "next/link";
import { Button } from "@/components/ui/button";
import { TextLink } from "@/components/ui/text-link";

export default function NotFound() {
  return (
    <div className="page-wrap py-24">
      <p className="eyebrow">404</p>
      <h1 className="display mt-4 text-4xl">This page is not here.</h1>
      <p className="mt-4 max-w-md text-base leading-relaxed text-ink-muted">
        The address may have moved, or the document you asked for is not in this demo.
      </p>
      <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3">
        <Button asChild>
          <Link href="/">Home</Link>
        </Button>
        <TextLink href="/demo">Open sample analysis</TextLink>
      </div>
    </div>
  );
}
