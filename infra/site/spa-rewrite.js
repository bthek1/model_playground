// CloudFront Function (cloudfront-js-2.0), viewer-request.
//
// SPA deep links: `/asr` or `/tasks/<slug>` is a route, not a file, so the
// bucket has no object for it. Any path whose last segment has no dot is
// served `/index.html` and the router takes it from there; a path with a dot
// (`/assets/x.js`, `/favicon.ico`, a `.wasm`) is fetched as itself, and a
// missing one fails rather than returning the app shell.
//
// This is deliberately not a distribution-wide custom error response: that
// would turn *every* 403/404 into index.html, including a future `/api`
// origin's. Consequence worth knowing: a route containing a dot would bypass
// the rewrite. No route has one; `spa-rewrite.test.ts` pins that.
//
// Plain ES5 with no exports — the runtime has no module system. The test
// evaluates this file's exact text.
function handler(event) {
  var request = event.request;
  var uri = request.uri;
  var last = uri.substring(uri.lastIndexOf("/") + 1);
  if (last.indexOf(".") === -1) {
    request.uri = "/index.html";
  }
  return request;
}
