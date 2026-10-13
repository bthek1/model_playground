/**
 * A task page's own words (#71): the sentence under its heading, and the
 * paragraphs that say what the page is in the vocabulary people search with.
 *
 * One file per page, imported by **two** places and no others: the page's own
 * route, which renders both (the tagline in the header, the paragraphs in an
 * "About" section below the workbench), and the build, which writes them into
 * that route's static HTML. Google indexes the page *after* rendering it, and
 * `createRoot` replaces the static `#root`, so text that lived only in the
 * static HTML would never be indexed by Google at all. Text that lived only in
 * the app would be invisible to every crawler that runs no JavaScript.
 *
 * Plain strings, never JSX: the build runs this in Node and escapes it.
 */
export interface PageCopy {
  /**
   * The first sentence of the page's header, 40–140 characters. Phrased the way
   * someone would describe what they want: "Turn one photo into a depth map,
   * on your own GPU."
   */
  tagline: string;
  /**
   * Two to four paragraphs (120–320 words in all): what the task is, which
   * models the page runs and how big they are, what runs where, what the page
   * cannot do, and that the input stays on the device. Every number comes from
   * the page's own catalogue — measured, never estimated.
   */
  about: readonly string[];
  /**
   * What people type when they want this page ("photo to depth map",
   * "monocular depth estimation"). Not a keyword list for the page — each one
   * must already appear in the title, tagline or `about`, and a test checks.
   */
  aliases: readonly string[];
}
