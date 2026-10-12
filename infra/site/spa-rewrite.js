// CloudFront Function (cloudfront-js-2.0), viewer-request.
//
// SPA deep links: `/asr` is a route, not a file, so the bucket has no object
// named for it. Any path whose last segment has no dot is a route; a path with
// a dot (`/assets/x.js`, `/favicon.ico`, a `.wasm`, `/robots.txt`) is fetched
// as itself, and a missing one fails rather than returning the app shell.
//
// A route is served one of three files. An indexable page has its own
// `/<route>/index.html`, written by the build with that page's title,
// description, canonical and a static summary (frontend/scripts/seoPages.ts,
// #64). A route the app has but search engines should not (APP_ROUTES:
// sign-in, sign-up) gets the root `/index.html` shell. Anything else gets
// `/404.html` — the same shell, so the app still renders its not-found page —
// which the deploy uploads with `x-amz-meta-status: 404`, and the
// viewer-response function (object-status.js) turns that into a real 404
// status (#63). A 200 for a URL with no page is a "soft 404" to a crawler.
//
// ROUTES must equal the frontend's SITE_PAGES, and
// frontend/src/seo/edgeRoutes.test.ts fails, naming the difference, when it
// does not: a page missing here would answer 404, and a route listed here with
// no file answers 403. spa-rewrite.test.ts checks that every route in the
// router is in ROUTES or APP_ROUTES, so a new route cannot 404 unnoticed.
//
// One URL per page (#63): `/` redirects to `/home`, and a trailing slash or an
// upper-case letter redirects to the lower-case, slash-free form, all with a
// 301, an absolute Location and the query string kept. Every route is
// lower-case kebab. The Location is absolute because Search Console reported a
// relative one (`/home`) as a "Redirect error", though it is valid HTTP.
//
// This is deliberately not a distribution-wide custom error response: that
// would turn *every* 403/404 into index.html, including a future `/api`
// origin's. Consequence worth knowing: a route containing a dot would bypass
// the rewrite. No route has one; `spa-rewrite.test.ts` pins that.
//
// Plain ES5 with no exports — the runtime has no module system. The test
// evaluates this file's exact text.
var APP_ROUTES = {
  "/login": true,
  "/signup": true
};

var ROUTES = {
  "/home": true,
  "/text-to-speech": true,
  "/asr": true,
  "/audio-to-audio": true,
  "/audio-classification": true,
  "/vad": true,
  "/depth": true,
  "/image-classification": true,
  "/object-detection": true,
  "/segmentation": true,
  "/super-resolution": true,
  "/video-classification": true,
  "/zero-shot-image-classification": true,
  "/mask-generation": true,
  "/zero-shot-object-detection": true,
  "/image-to-3d": true,
  "/image-features": true,
  "/pose": true,
  "/background-removal": true,
  "/image-text-to-text": true,
  "/visual-question-answering": true,
  "/video-text-to-text": true,
  "/text-classification": true,
  "/token-classification": true,
  "/question-answering": true,
  "/zero-shot-classification": true,
  "/translation": true,
  "/summarization": true,
  "/text-features": true,
  "/text-generation": true,
  "/fill-mask": true,
  "/sentence-similarity": true,
  "/text-ranking": true,
  "/graph": true,
  "/link-prediction": true,
  "/graph-classification": true,
  "/rl": true,
  "/robotics": true,
  "/tabular-classification": true,
  "/tabular-regression": true,
  "/time-series-forecasting": true,
  "/training": true,
  "/discrete-maths": true,
  "/tensor": true,
  "/playground": true,
  "/privacy": true,
  "/terms": true,
  "/licences": true,
  "/accessibility": true
};

function redirect(request, path) {
  var querystring = request.querystring;
  var host = request.headers && request.headers.host && request.headers.host.value;
  var location = host ? "https://" + host + path : path;
  var pairs = [];
  for (var key in querystring) {
    var entry = querystring[key];
    if (entry.multiValue) {
      for (var i = 0; i < entry.multiValue.length; i++) {
        pairs.push(key + "=" + entry.multiValue[i].value);
      }
    } else {
      pairs.push(entry.value === "" ? key : key + "=" + entry.value);
    }
  }
  return {
    statusCode: 301,
    statusDescription: "Moved Permanently",
    headers: {
      location: { value: pairs.length ? location + "?" + pairs.join("&") : location },
    },
  };
}

function handler(event) {
  var request = event.request;
  var uri = request.uri;
  var last = uri.substring(uri.lastIndexOf("/") + 1);
  if (last.indexOf(".") !== -1) {
    return request;
  }
  var path = uri.toLowerCase().replace(/\/+$/, "");
  if (path === "") {
    return redirect(request, "/home");
  }
  if (path !== uri) {
    return redirect(request, path);
  }
  if (ROUTES[path] === true) {
    request.uri = path + "/index.html";
  } else if (APP_ROUTES[path] === true) {
    request.uri = "/index.html";
  } else {
    request.uri = "/404.html";
  }
  return request;
}
