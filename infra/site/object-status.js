// CloudFront Function (cloudfront-js-2.0), viewer-response.
//
// Lets an object in the bucket choose its own HTTP status (#63). S3 always
// answers 200 for an object it has, so `/404.html` — the app shell that
// spa-rewrite.js serves for a path with no page — would reach the viewer as a
// 200: a "soft 404", which search engines index as a duplicate of the shell.
// The deploy uploads that one object with `--metadata status=404`; S3 returns
// it as `x-amz-meta-status`, and this function makes it the status line and
// drops the header.
//
// Keyed on the object, not on the URL, deliberately: whether a viewer-response
// function sees the URI before or after the viewer-request rewrite is not
// something to depend on, and the metadata travels with the bytes either way.
// Only a 4xx is honoured, so a mistyped value cannot turn a page into a
// redirect or an error the browser will not render.
//
// Plain ES5 with no exports — the runtime has no module system. The test
// evaluates this file's exact text.
function handler(event) {
  var response = event.response;
  var header = response.headers["x-amz-meta-status"];
  if (!header) {
    return response;
  }
  delete response.headers["x-amz-meta-status"];
  var status = parseInt(header.value, 10);
  if (status >= 400 && status <= 499) {
    response.statusCode = status;
    response.statusDescription = status === 404 ? "Not Found" : "Error";
  }
  return response;
}
