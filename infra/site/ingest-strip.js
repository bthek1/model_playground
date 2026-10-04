// CloudFront Function (cloudfront-js-2.0), viewer-request, on `/ingest/*` only.
//
// Same-origin analytics (#60): the app sends PostHog events to
// `https://<site>/ingest/...`, and this behaviour forwards them to PostHog's
// ingestion host. PostHog serves `/e/`, `/i/v0/e/`, `/flags/`, … at its root,
// so the `/ingest` prefix has to go — CloudFront's origin path can only *add*
// a prefix, never remove one.
//
// It must NOT be `spa-rewrite.js`: that one turns every extensionless path into
// `/index.html`, so `/ingest/e/` would be answered with the app shell and a
// 200, and every event would vanish without an error. `site.test.ts` pins
// which function sits on which behaviour.
//
// Plain ES5 with no exports — the runtime has no module system. The test
// evaluates this file's exact text.
function handler(event) {
  var request = event.request;
  var uri = request.uri;
  if (uri === "/ingest" || uri.indexOf("/ingest/") === 0) {
    request.uri = uri.substring("/ingest".length) || "/";
  }
  return request;
}
