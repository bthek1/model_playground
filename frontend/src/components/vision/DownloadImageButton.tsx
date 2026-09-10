// "Download PNG", for the two routes that produce a picture the user wants to
// keep — `/background-removal` and `/super-resolution`.
//
// Small, and three things about it are easy to get wrong:
//
//  1. **PNG, never JPEG.** JPEG has no alpha channel, so a "download the
//     cut-out" button that produced one would hand back the subject on an
//     arbitrary opaque background — the single thing the matting page exists to
//     avoid.
//  2. **`nativeButton={false}`.** Base UI's `Button` assumes a real `<button>`
//     unless told otherwise, and warns at runtime when `render` supplies
//     anything else. A download needs an `<a download>`, so the flag is
//     required rather than optional.
//  3. **The object URL is revoked.** It is created from a blob of the full-size
//     image — megabytes — and one per result, so leaving them alive pins every
//     picture the user has produced for the tab's lifetime.
//
// While the blob is still encoding there is no `href`. An `<a>` without one is
// not a link and is not focusable, so it is already inert; `aria-disabled` says
// so out loud rather than relying on that.

import { Download } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import type { Pixels } from "@/vision/draw";
import { toPngBlob } from "@/vision/matte";

export function DownloadImageButton({
  image,
  filename,
  label = "Download PNG",
}: {
  image: Pixels;
  filename: string;
  label?: string;
}) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    let created: string | null = null;

    void toPngBlob(image).then((blob) => {
      if (!blob) return;
      created = URL.createObjectURL(blob);
      if (live) setUrl(created);
      else URL.revokeObjectURL(created);
    });

    return () => {
      live = false;
      setUrl(null);
      if (created) URL.revokeObjectURL(created);
    };
  }, [image]);

  return (
    <Button
      size="sm"
      variant="outline"
      nativeButton={false}
      render={
        <a
          href={url ?? undefined}
          download={filename}
          aria-disabled={url == null}
          data-testid="download-png"
        >
          <Download className="size-4" /> {url ? label : "Preparing PNG…"}
        </a>
      }
    />
  );
}
