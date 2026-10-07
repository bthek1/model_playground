// The privacy notice (#62). Every list on it is rendered from data the code
// itself uses — `legal/recipients.ts`, `legal/storage.ts` — and gated on the
// same build flags as the behaviour it describes, so a build without analytics
// never claims to send any, and one without a backend never mentions accounts.
// A sentence describing a request the build does not make would be as untrue
// as the reverse.

import { createLazyFileRoute, Link } from "@tanstack/react-router";

import { ANALYTICS_NOTE } from "@/components/analytics/AnalyticsNote";
import {
  Cell,
  ContactLink,
  ExternalLink,
  LegalPage,
  LegalTable,
  Section,
} from "@/components/legal/LegalPage";
import { RECIPIENTS } from "@/legal/recipients";
import { JURISDICTION, OPERATOR } from "@/legal/site";
import { STORED_ITEMS } from "@/legal/storage";
import { ANALYTICS_ENABLED, BACKEND_ENABLED } from "@/lib/features";

export const Route = createLazyFileRoute("/privacy")({
  component: PrivacyPage,
});

export function PrivacyPage() {
  const recipients = RECIPIENTS.filter((r) => ANALYTICS_ENABLED || !r.analyticsOnly);
  const stored = STORED_ITEMS.filter(
    (s) => (BACKEND_ENABLED || !s.backendOnly) && (ANALYTICS_ENABLED || !s.analyticsOnly),
  );

  return (
    <LegalPage
      title="Privacy"
      intro={
        <>
          <p>
            This site is run by {OPERATOR}, an individual based in {JURISDICTION}, as a free,
            non-commercial project. Questions or requests about your data go to <ContactLink />.
          </p>
          <p className="font-medium text-foreground">
            The models on this site run in your browser. What you give them (text you type,
            files you open, pictures, audio, your camera and microphone) is processed on your
            device and is never sent to this site or to anyone else.
          </p>
        </>
      }
    >
      <Section id="device" title="What stays on your device">
        <p>
          Inference, training and every result happen inside the browser tab. A file you open is
          read by the page, not uploaded. Camera and microphone access is asked for by your
          browser, used only while the page is open, and released when you stop. Closing or
          reloading the tab discards your inputs and results.
        </p>
      </Section>

      <Section id="recipients" title="Who receives anything, and what">
        <p>
          Loading a web page, and downloading a model, are ordinary HTTP requests. Whoever serves
          the file sees the request: your IP address, your browser&apos;s user agent and which
          file was asked for. These are the services involved:
        </p>
        <LegalTable
          caption="Services that receive requests from your browser"
          head={["Service", "What for", "What it receives", "Where"]}
          testId="privacy-recipients"
        >
          {recipients.map((r) => (
            <tr key={r.id} data-recipient={r.id}>
              <Cell className="font-medium">
                <ExternalLink href={r.policyUrl}>{r.name}</ExternalLink>
              </Cell>
              <Cell>{r.role}</Cell>
              <Cell>{r.receives}</Cell>
              <Cell>{r.location}</Cell>
            </tr>
          ))}
        </LegalTable>
        <p>
          Each service handles that information under its own privacy policy, linked above.
          This site sells nothing, shows no ads and shares nothing with anyone else.
        </p>
      </Section>

      <Section id="analytics" title="Analytics">
        {ANALYTICS_ENABLED ? (
          <>
            <p data-testid="privacy-analytics">
              {ANALYTICS_NOTE} Events record what the app did (which page, which model, how long
              it took, whether WebGPU was available) and never what you gave it. They carry no
              cookie, no device identifier and nothing stored on your device, so a returning
              visit cannot be linked to an earlier one.
            </p>
            <p>
              You can turn analytics off with the switch in the system panel. Once it is off,
              the analytics code is not even downloaded.
            </p>
          </>
        ) : (
          <p data-testid="privacy-analytics">This build of the site sends no analytics.</p>
        )}
      </Section>

      {BACKEND_ENABLED && (
        <Section id="accounts" title="Accounts">
          <p data-testid="privacy-accounts">
            This build offers optional sign-in. If you create an account, your email address,
            name and a hash of your password are stored on this site&apos;s server, together with
            any inference-run records you choose to save. They are used only to provide the
            account. Ask at <ContactLink /> to have the account deleted.
          </p>
        </Section>
      )}

      <Section id="storage" title="What the site stores in your browser">
        <p>
          No cookies are set. The site keeps the items below in your browser&apos;s own storage,
          each one either a choice you made or something you asked it to download. None is sent
          anywhere. Clearing this site&apos;s data in your browser removes all of them.
        </p>
        <LegalTable
          caption="Items stored in your browser"
          head={["Name", "Kind", "What it holds"]}
          testId="privacy-storage"
        >
          {stored.map((s) => (
            <tr key={s.key} data-key={s.key}>
              <Cell>
                <code>{s.key}</code>
              </Cell>
              <Cell>{s.where}</Cell>
              <Cell>{s.purpose}</Cell>
            </tr>
          ))}
        </LegalTable>
      </Section>

      <Section id="basis" title="Why, and for how long">
        <p>
          Serving the site and counting its use anonymously are in the site&apos;s legitimate
          interest in running and improving a free tool (GDPR Art. 6(1)(f), for visitors from
          the EU or UK). The site itself keeps no server logs.{" "}
          {ANALYTICS_ENABLED &&
            "Analytics events are kept by PostHog under its retention settings. Because they contain nothing that identifies you, they cannot be looked up or deleted per person. "}
          Everything stored in your browser stays until you clear it.
        </p>
        <p>
          Hugging Face, Google{ANALYTICS_ENABLED && " and PostHog"} are based in the United
          States, so a request your browser makes to them may be processed outside Australia
          and outside the EU or UK.
        </p>
      </Section>

      <Section id="rights" title="Your rights">
        <p>
          Under the Australian Privacy Principles, and the GDPR or UK GDPR where they apply, you
          can ask what personal information is held about you and ask for it to be corrected or
          deleted, and you can object to its use. Write to <ContactLink />. In practice the site
          holds nothing that identifies you, so most requests can be answered by saying so.
        </p>
        <p>
          If you are unhappy with the answer, you can complain to the{" "}
          <ExternalLink href="https://www.oaic.gov.au/privacy/privacy-complaints">
            Office of the Australian Information Commissioner
          </ExternalLink>{" "}
          or to the data protection authority where you live.
        </p>
      </Section>

      <Section id="changes" title="Changes">
        <p>
          If what the site collects changes, this page changes with it, and the date at the top
          moves. The <Link to="/terms" className="underline underline-offset-2">terms of use</Link>{" "}
          and the <Link to="/licences" className="underline underline-offset-2">licences</Link> are
          separate pages.
        </p>
      </Section>
    </LegalPage>
  );
}
