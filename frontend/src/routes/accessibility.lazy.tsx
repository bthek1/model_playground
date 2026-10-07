// The accessibility statement (#62). Not a legal requirement for a free,
// non-commercial site (the European Accessibility Act exempts it), but the
// target is WCAG 2.2 AA anyway, and naming the known gaps is more useful than
// claiming none.

import { createLazyFileRoute } from "@tanstack/react-router";

import { ContactLink, LegalPage, Section } from "@/components/legal/LegalPage";

export const Route = createLazyFileRoute("/accessibility")({
  component: AccessibilityPage,
});

function AccessibilityPage() {
  return (
    <LegalPage
      title="Accessibility"
      intro={
        <p>
          The aim is for every page to meet the Web Content Accessibility Guidelines (WCAG) 2.2 at
          level AA. That aim is not yet fully met. Here is what is done, and what is known not to
          work.
        </p>
      }
    >
      <Section id="done" title="What is in place">
        <ul className="list-disc space-y-1 pl-5">
          <li>
            Every model page uses the same four labelled regions (choose, load, run, output),
            in the same order, at every screen width.
          </li>
          <li>
            Controls are native buttons and form fields, reachable and operable by keyboard.
          </li>
          <li>
            A licence restriction, an entity type or a warning is stated in words, never by
            colour alone.
          </li>
          <li>
            The graph, overlay and reinforcement-learning canvases carry a text description,
            and the reinforcement learning page renders its learned policy as text as well as
            arrows.
          </li>
          <li>Light and dark themes both follow your system setting by default.</li>
        </ul>
      </Section>

      <Section id="gaps" title="Known limitations">
        <ul className="list-disc space-y-1 pl-5">
          <li>
            Interactive canvases (the graph layouts, the point cloud, the mask-generation
            picker) need a pointer. Their results are described in text, but the interaction
            itself has no keyboard equivalent.
          </li>
          <li>
            Most charts (loss curves, backtests, residual plots) have no text alternative yet.
            The key numbers beside them are text.
          </li>
          <li>
            Most models need a WebGPU-capable browser. The fallback runs on the CPU, but it can
            be slow enough that progress announcements are the only feedback for a long time.
          </li>
        </ul>
      </Section>

      <Section id="contact" title="Tell us">
        <p>
          If something on the site does not work with your assistive technology, write to{" "}
          <ContactLink /> and say which page and what you were trying to do.
        </p>
      </Section>
    </LegalPage>
  );
}
