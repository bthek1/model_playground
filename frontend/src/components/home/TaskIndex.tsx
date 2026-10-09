// Every task the site has, as plain links with one line each (#65).
//
// The sidebar lists the same tasks but starts collapsed, so on a fresh visit
// neither a person nor a crawler sees more than the category names. This is
// the page that links to every task page in body text, from the same taxonomy
// entries — and the same descriptions — the build writes into each route's
// meta tags. Each category is an anchor (`/home#audio`) that the per-route
// breadcrumbs and the structured data point at.

import { Link } from "@tanstack/react-router";

import { taskCategories } from "@/components/layout/taskTaxonomy";
import { categoryAnchor } from "@/seo/pages";

export function TaskIndex() {
  return (
    <section aria-labelledby="task-index-heading" className="space-y-6">
      <h2 id="task-index-heading" className="text-lg font-semibold">
        What you can run
      </h2>
      {taskCategories.map((category) => (
        <section
          key={category.label}
          id={categoryAnchor(category.label)}
          aria-labelledby={`${categoryAnchor(category.label)}-heading`}
          className="scroll-mt-4 space-y-2"
        >
          <h3
            id={`${categoryAnchor(category.label)}-heading`}
            className="flex items-center gap-2 text-sm font-medium"
          >
            <category.icon className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
            {category.label}
          </h3>
          <ul className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {category.tasks.map((t) => (
              <li key={t.slug} className="text-sm">
                <Link
                  to={t.to}
                  className="font-medium underline-offset-2 hover:underline"
                >
                  {t.label}
                </Link>
                <p className="text-muted-foreground">{t.description}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </section>
  );
}
