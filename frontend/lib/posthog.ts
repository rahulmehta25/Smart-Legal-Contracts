"use client";
import posthog from "posthog-js";

const POSTHOG_KEY =
  process.env.NEXT_PUBLIC_POSTHOG_KEY ?? "phc_MdeB4wNRGn9gckJkl0ZZDh1XEIlvsfWBa23qv9lty5Z";
const POSTHOG_HOST = process.env.NEXT_PUBLIC_POSTHOG_HOST ?? "https://us.i.posthog.com";

let initialized = false;

export function initPostHog() {
  if (initialized || typeof window === "undefined") return;
  posthog.init(POSTHOG_KEY, {
    api_host: POSTHOG_HOST,
    person_profiles: "always",
    capture_pageview: true,
    capture_pageleave: true,
    autocapture: false,
  });
  initialized = true;
}

export { posthog };
