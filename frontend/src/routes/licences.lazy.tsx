// Attribution for every model, dataset and package the site uses (#62).
//
// Generated, never hand-maintained: the models come from `MODEL_CATALOGUES`
// through the licence registry, the datasets from `legal/datasets.ts` and the
// tabular/forecast catalogues, and the packages from the THIRD-PARTY-NOTICES
// file the build writes. A model added to any catalogue appears here with no
// edit to this file, and `licences.test.ts` fails if it has no licence row.

import { createLazyFileRoute } from "@tanstack/react-router";

import {
  Cell,
  ContactLink,
  ExternalLink,
  LegalPage,
  LegalTable,
  Section,
} from "@/components/legal/LegalPage";
import { FORECAST_SAMPLES } from "@/forecast/samples";
import { MODEL_CATALOGUES } from "@/legal/catalogues";
import { DATASETS } from "@/legal/datasets";
import { SOURCE_URL } from "@/legal/site";
import { entryRepos, licenceFor, TERMS_LABEL, type ModelLicence } from "@/model/licences";
import { SAMPLES as TABULAR_SAMPLES } from "@/tabular/samples";
import { cn } from "@/lib/utils";

export const Route = createLazyFileRoute("/licences")({
  component: LicencesPage,
});

/** The file the build writes beside index.html (`scripts/third-party-notices.mjs`). */
export const NOTICES_PATH = "/THIRD-PARTY-NOTICES.txt";

interface ModelRow {
  repo: string;
  labels: Set<string>;
  pages: Set<string>;
  licence: ModelLicence;
}

/** One row per repo, however many pages and entries offer it. */
export function modelRows(): ModelRow[] {
  const rows = new Map<string, ModelRow>();
  for (const c of MODEL_CATALOGUES) {
    for (const entry of c.entries) {
      for (const repo of entryRepos(entry)) {
        const licence = licenceFor(repo);
        if (!licence) continue;
        const row = rows.get(repo) ?? { repo, labels: new Set(), pages: new Set(), licence };
        row.labels.add(entryRepos(entry).length > 1 ? repo.split("/")[1] : entry.label);
        c.pages.forEach((p) => row.pages.add(p));
        rows.set(repo, row);
      }
    }
  }
  return [...rows.values()].sort((a, b) => a.repo.localeCompare(b.repo));
}

function Terms({ terms }: { terms: ModelLicence["terms"] }) {
  return (
    <span
      className={cn(
        "whitespace-nowrap",
        terms !== "permissive" && "font-medium text-amber-600 dark:text-amber-500",
      )}
    >
      {TERMS_LABEL[terms]}
    </span>
  );
}

function LicencesPage() {
  const rows = modelRows();
  const restricted = rows.filter((r) => r.licence.terms !== "permissive").length;

  return (
    <LegalPage
      title="Licences"
      dated={false}
      intro={
        <p>
          Every model, dataset and open-source package this site uses, with its licence. The
          models and datasets are not hosted here: your browser downloads them from their
          publishers. {rows.length} models are listed. {restricted} of them are not permissive,
          and their pickers say so before you load anything. Corrections are welcome at{" "}
          <ContactLink />.
        </p>
      }
    >
      <Section id="models" title="Models">
        <LegalTable
          caption="Models, their licences and the pages that offer them"
          head={["Model", "Licence", "Terms", "Used on"]}
          testId="licences-models"
        >
          {rows.map((r) => (
            <tr key={r.repo} data-repo={r.repo} data-terms={r.licence.terms}>
              <Cell>
                <ExternalLink href={`https://huggingface.co/${r.repo}`}>{r.repo}</ExternalLink>
                {r.licence.from && (
                  <span className="block text-muted-foreground">
                    converted from{" "}
                    <ExternalLink href={`https://huggingface.co/${r.licence.from}`}>
                      {r.licence.from}
                    </ExternalLink>
                  </span>
                )}
              </Cell>
              <Cell>
                <ExternalLink href={r.licence.url}>{r.licence.name}</ExternalLink>
                {(r.licence.note ?? r.licence.via) && (
                  <span className="block text-muted-foreground">
                    {r.licence.note ?? r.licence.via}
                  </span>
                )}
              </Cell>
              <Cell>
                <Terms terms={r.licence.terms} />
              </Cell>
              <Cell className="text-muted-foreground">{[...r.pages].join(", ")}</Cell>
            </tr>
          ))}
        </LegalTable>
      </Section>

      <Section id="datasets" title="Datasets and samples">
        <LegalTable
          caption="Datasets and sample inputs, with their licences"
          head={["Dataset", "Source", "Licence", "Used on"]}
          testId="licences-datasets"
        >
          {DATASETS.map((d) => (
            <tr key={d.id} data-dataset={d.id}>
              <Cell className="font-medium">{d.name}</Cell>
              <Cell className="text-muted-foreground">{d.credit}</Cell>
              <Cell>
                <ExternalLink href={d.url}>{d.licence}</ExternalLink>
              </Cell>
              <Cell className="text-muted-foreground">{d.pages.join(", ")}</Cell>
            </tr>
          ))}
          {TABULAR_SAMPLES.map((s) => (
            <tr key={`tabular-${s.id}`} data-dataset={`tabular-${s.id}`}>
              <Cell className="font-medium">{s.label}</Cell>
              <Cell className="text-muted-foreground">{s.source}</Cell>
              <Cell>{s.licence}</Cell>
              <Cell className="text-muted-foreground">/tabular-classification, /tabular-regression</Cell>
            </tr>
          ))}
          {FORECAST_SAMPLES.map((s) => (
            <tr key={`forecast-${s.id}`} data-dataset={`forecast-${s.id}`}>
              <Cell className="font-medium">{s.label}</Cell>
              <Cell className="text-muted-foreground">{s.source}</Cell>
              <Cell>{s.licence}</Cell>
              <Cell className="text-muted-foreground">/time-series-forecasting</Cell>
            </tr>
          ))}
        </LegalTable>
      </Section>

      <Section id="software" title="Software">
        <p>
          The site&apos;s own code is MIT-licensed:{" "}
          <ExternalLink href={SOURCE_URL}>{SOURCE_URL.replace("https://", "")}</ExternalLink>. It
          is built on open-source packages, including React, TanStack Router and Query, Base UI,
          Transformers.js and ONNX Runtime Web. Their copyright and licence notices, as their
          licences require, are in{" "}
          <a
            href={NOTICES_PATH}
            data-testid="third-party-notices"
            className="underline underline-offset-2"
          >
            THIRD-PARTY-NOTICES.txt
          </a>
          , generated from exactly what this build ships.
        </p>
      </Section>
    </LegalPage>
  );
}
