// Who builds the site, and how its claims are checked (#72). Search engines
// and readers both look for the person behind a page; until this one nothing
// on the site said. Facts come from the same modules the legal pages read
// (`legal/site.ts`), and the structured data naming the author is built from
// them too (`seo/render.ts`), so the page and the JSON-LD cannot disagree.

import { createLazyFileRoute, Link } from "@tanstack/react-router";

import {
  ContactLink,
  ExternalLink,
  LegalPage,
  Section,
} from "@/components/legal/LegalPage";
import { AUTHOR_URL, JURISDICTION, OPERATOR, SOURCE_URL } from "@/legal/site";

export const Route = createLazyFileRoute("/about")({
  component: AboutSitePage,
});

const inline = "underline underline-offset-2";

export function AboutSitePage() {
  return (
    <LegalPage
      title="About"
      dated={false}
      intro={
        <p>
          Model Playground is built and maintained by {OPERATOR}, an individual based in{" "}
          {JURISDICTION}, as a free, non-commercial project. Every model on it runs in your
          browser, on your own GPU or CPU, and every page says what it runs, how big the download
          is and what the model cannot do.
        </p>
      }
    >
      <Section id="who" title="Who builds it">
        <p>
          {OPERATOR} writes the code, chooses the models and writes the pages. The code is on{" "}
          <ExternalLink href={AUTHOR_URL}>GitHub</ExternalLink>, and questions, corrections and bug
          reports go to <ContactLink />.
        </p>
      </Section>

      <Section id="why" title="Why it exists">
        <p>
          To show what machine learning can do on the hardware already in front of you, and how
          each model actually behaves: its size, its precision, where it runs and how it fails.
          Running in the browser is also a privacy property, not only a performance one. Text you
          redact, audio you transcribe and a CSV you fit a model to are processed on your device
          and never uploaded (<Link to="/privacy" className={inline}>privacy</Link>).
        </p>
      </Section>

      <Section id="models" title="How models are chosen">
        <p>A task gets a page only if it clears two bars:</p>
        <ul className="list-disc space-y-1 pl-5">
          <li>It runs entirely client-side, on WebGPU or WebAssembly, with no server.</li>
          <li>
            Its cheapest usable checkpoint is under about 500 MB. Sizes are measured from the
            files on the Hugging Face Hub, never estimated from a parameter count, and re-checked
            by an automated test.
          </li>
        </ul>
        <p>
          Three pages were built, worked, and were then removed because every checkpoint for
          their task was a several-hundred-megabyte download. Where a page pins a model to a
          particular precision, it records whether that pin is a measurement or a precaution.
        </p>
      </Section>

      <Section id="checks" title="How the pages are checked">
        <ul className="list-disc space-y-1 pl-5">
          <li>
            Each model page has an end-to-end test that downloads the real weights and asserts a
            known answer on a known input (a label on an image, a transcript of a clip), because
            &ldquo;a result appeared&rdquo; also passes when a quantized model is wrong.
          </li>
          <li>
            Signal processing and environments are pinned against their reference
            implementations, not against expectations: DeepFilterNet against libDF, CartPole
            against a Gymnasium trajectory, the karate-club graph against networkx.
          </li>
          <li>
            Every gradient written by hand is checked by finite differences, and the
            reinforcement-learning page checks its learned table against an exact solution.
          </li>
          <li>
            Where a score has a trivial baseline, such as always guessing the most common class,
            the baseline is shown beside it.
          </li>
        </ul>
      </Section>

      <Section id="source" title="Source and licences">
        <p>
          The source code is public at{" "}
          <ExternalLink href={SOURCE_URL}>github.com/bthek1/model_playground</ExternalLink>. Every
          model, dataset and package the site uses is listed with its licence on the{" "}
          <Link to="/licences" className={inline}>licences</Link> page, with non-commercial terms
          marked where they apply.
        </p>
      </Section>
    </LegalPage>
  );
}
