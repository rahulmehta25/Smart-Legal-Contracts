import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center border px-2 py-0.5 text-[0.65rem] font-medium uppercase tracking-[0.14em] transition-colors duration-150",
  {
    variants: {
      variant: {
        default: "border-ink/15 bg-transparent text-ink",
        secondary: "border-ink/10 bg-linen text-ink-muted",
        destructive: "border-transparent bg-destructive text-destructive-foreground",
        outline: "text-foreground",
        success: "border-emerald-800/20 bg-transparent text-emerald-900",
        warning: "border-amber-800/20 bg-transparent text-amber-900",
        danger: "border-red-800/20 bg-transparent text-red-900",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
