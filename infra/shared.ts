// What the two stacks must agree on without reading each other's state.
//
// The bootstrap stack writes IAM policies for resources the site stack has not
// created yet, so it cannot look them up; both sides take the names from here
// instead. A mismatch fails loudly (AccessDenied in CI), but it is cheaper not
// to have one.

/** Every resource the site stack owns carries it; the deploy role's
 *  destructive actions on shared-namespace resources (ACM certificates,
 *  CloudFront distributions) are conditioned on it. */
export const PROJECT_TAG = { key: "project", value: "model-playground" } as const;

/** CloudFront Function names are account-global and appear in their ARN, so
 *  the deploy role can be scoped by name. */
export const FUNCTION_PREFIX = "model-playground-";

/** AWS's managed `CachingOptimized` cache policy. Its ID is fixed and
 *  documented; using it directly spares the deploy role `ListCachePolicies`. */
export const CACHING_OPTIMIZED_POLICY_ID = "658327ea-f89d-4fab-a63d-7e88639e58f6";

/** CloudFront reads viewer certificates only from here. */
export const CERT_REGION = "us-east-1";

/** AWS's managed `CachingDisabled` cache policy — for `/ingest/*` (#60): an
 *  analytics POST must reach PostHog every time, never be answered from cache. */
export const CACHING_DISABLED_POLICY_ID = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad";

/** AWS's managed `AllViewerExceptHostHeader` origin request policy. Forwards the
 *  query string, cookies and headers the SDK sends, but lets CloudFront set
 *  `Host` to the origin's own name — PostHog's ingestion routes on it. */
export const ALL_VIEWER_EXCEPT_HOST_POLICY_ID = "b689b0a8-53d0-40ab-baf2-68738e2966ac";
