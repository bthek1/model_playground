// Who runs the site and how to reach them — the facts every legal page states
// (#62). Written once so a change of address or a revision date cannot leave
// one page disagreeing with another.

/** The person responsible for the site: the "controller", in GDPR's terms. */
export const OPERATOR = "Benedict Thekkel";

/**
 * Where privacy, copyright and licence questions go. A dedicated alias rather
 * than a personal inbox, because every page below publishes it.
 *
 * **It must exist before the site is deployed** — a privacy notice whose only
 * contact bounces is worse than none.
 */
export const CONTACT_EMAIL = "privacy@benedictthekkel.com";

/** The public origin the static build is served from (#57). */
export const SITE_ORIGIN = "https://playground.benedictthekkel.com";

/** Where the operator is based: the governing law, and the regulator named. */
export const JURISDICTION = "Australia";

/**
 * The date the privacy notice and terms last changed in substance. Bump it
 * with any edit a visitor would want to know about.
 */
export const LAST_UPDATED = "2026-10-07";

/** The source repository, for the code's own licence. */
export const SOURCE_URL = "https://github.com/bthek1/model_playground";
