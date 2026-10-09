// CloudFront Function (cloudfront-js-2.0), viewer-request.
//
// SPA deep links: `/asr` is a route, not a file, so the bucket has no object
// named for it. Any path whose last segment has no dot is a route; a path with
// a dot (`/assets/x.js`, `/favicon.ico`, a `.wasm`, `/robots.txt`) is fetched
// as itself, and a missing one fails rather than returning the app shell.
//
// A route is served one of two files (#64). An indexable page has its own
// `/<route>/index.html`, written by the build with that page's title,
// description, canonical and a static summary (frontend/scripts/seoPages.ts);
// anything else — sign-in, a mistyped URL — gets the root `/index.html` shell,
// and the app's not-found page says `noindex`. ROUTES below is the list of
// pages that have a file. It must equal the frontend's SITE_PAGES, and
// frontend/src/seo/edgeRoutes.test.ts fails, naming the difference, when it
// does not: a route missing here is served the generic shell, and a route
// listed here with no file answers 403.
//
// One URL per page (#63): `/` redirects to `/home`, and a trailing slash or an
// upper-case letter redirects to the lower-case, slash-free form, all with a
// 301 and the query string kept. Every route is lower-case kebab.
//
// This is deliberately not a distribution-wide custom error response: that
// would turn *every* 403/404 into index.html, including a future `/api`
// origin's. Consequence worth knowing: a route containing a dot would bypass
// the rewrite. No route has one; `spa-rewrite.test.ts` pins that.
//
// Plain ES5 with no exports — the runtime has no module system. The test
// evaluates this file's exact text.
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

function redirect(location, querystring) {
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
    return redirect("/home", request.querystring);
  }
  if (path !== uri) {
    return redirect(path, request.querystring);
  }
  request.uri = ROUTES[path] === true ? path + "/index.html" : "/index.html";
  return request;
}
