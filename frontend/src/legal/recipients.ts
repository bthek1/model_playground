// Every third party a visitor's browser talks to, and what each one learns
// (#62). The privacy notice renders this list; `privacy.test.tsx` asserts each
// one is named on the page.
//
// The rule that keeps it short is architectural: inference runs in the tab, so
// nothing a visitor types, uploads, records or films is ever sent anywhere.
// What does leave is the ordinary metadata of an HTTP request — an IP address,
// a user agent, which file was asked for — to whoever serves the file.
//
// A new host the browser fetches from (a CDN, a dataset mirror, a font) is a
// new recipient and needs a row here. `e2e/specs/legal.spec.ts` cannot find it
// for you; the static-build spec only proves `/api` is never called.

export interface Recipient {
  id: string;
  name: string;
  /** What it does for the site. */
  role: string;
  /** What it receives, and when. */
  receives: string;
  /** Where it processes data. */
  location: string;
  policyUrl: string;
  /** Only contacted by a build that ships analytics. */
  analyticsOnly?: boolean;
}

export const RECIPIENTS: readonly Recipient[] = [
  {
    id: "aws",
    name: "Amazon Web Services (CloudFront and S3)",
    role: "Hosts and serves this site.",
    receives:
      "Each request for a page or script: your IP address, browser user agent and the URL requested. Access logging is switched off, so the site keeps no record of these requests.",
    location: "CloudFront's nearest edge location; the files are stored in Sydney, Australia.",
    policyUrl: "https://aws.amazon.com/privacy/",
  },
  {
    id: "huggingface",
    name: "Hugging Face",
    role: "Hosts the model weights, the sample media and the PROTEINS dataset. Your browser downloads them directly; this site never proxies them.",
    receives:
      "When you press Load, pick a sample, or open the graph-classification page's dataset: your IP address, browser user agent and the files requested, which identify the model.",
    location: "United States, served through its CDN.",
    policyUrl: "https://huggingface.co/privacy",
  },
  {
    id: "google",
    name: "Google Cloud Storage",
    role: "Hosts the MNIST digits used by the linear-model training page.",
    receives:
      "When you load that page's dataset (once, then it is cached in your browser): your IP address, browser user agent and the file requested.",
    location: "Google's global infrastructure.",
    policyUrl: "https://policies.google.com/privacy",
  },
  {
    id: "posthog",
    name: "PostHog",
    role: "Anonymous product analytics: which pages are opened and which models are loaded.",
    receives:
      "Events about what the app did: the page's route pattern, the model id and backend, load and run times, whether WebGPU is available, and an error category. Never your inputs, files, outputs or any text you typed. Requests go through this site's own address, which forwards your IP address; PostHog is configured to discard it. No cookie or device storage is used, so no profile of you is built.",
    location: "United States.",
    policyUrl: "https://posthog.com/privacy",
    analyticsOnly: true,
  },
];
