import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="page-wrap py-16">
      <div className="space-y-6">
        <Skeleton className="h-8 w-64 bg-linen" />
        <Skeleton className="h-4 w-96 bg-linen" />
        <div className="grid md:grid-cols-3 gap-4">
          <Skeleton className="h-24 bg-linen" />
          <Skeleton className="h-24 bg-linen" />
          <Skeleton className="h-24 bg-linen" />
        </div>
        <Skeleton className="h-64 bg-linen" />
      </div>
    </div>
  );
}
