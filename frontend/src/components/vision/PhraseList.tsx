// An editable list of short strings, for the three routes that ask the user to
// supply the model's vocabulary: the labels on /zero-shot-image-classification,
// the queries on /zero-shot-object-detection, and the labels again on
// /video-classification.
//
// Extracted at the second caller rather than the third, because the accessible
// wiring is the part that gets quietly dropped on a copy: the text field is
// *labelled by* the field name (so `getByLabelText(/labels/)` finds it), each
// chip's remove button carries its own name (so "Remove dog" is unambiguous with
// twelve chips on screen), and Enter adds without submitting anything.
//
// It is a controlled component with a single `onChange` — the draft text is the
// only state it owns, because a half-typed word is not part of the page's model
// and should not survive a re-render of the route around it.

import { Plus, X } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function PhraseList({
  id,
  title,
  items,
  onChange,
  placeholder,
  disabled = false,
  emptyHint = "Add at least one.",
  hint,
}: {
  /** DOM id for the text field; must be unique on the page. */
  id: string;
  /** The field's visible name — also what `getByLabelText` will match. */
  title: string;
  items: readonly string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  /** Shown in place of the chips when the list is empty. */
  emptyHint?: string;
  /** One line under the field: what these particular strings are for. */
  hint?: React.ReactNode;
}) {
  const [draft, setDraft] = useState("");

  const add = () => {
    const value = draft.trim();
    // Silently ignoring a duplicate is deliberate: the model would score the
    // same string twice and the second row would be noise, but a rejection
    // message for typing a word already on screen is worse than nothing.
    if (!value || items.includes(value)) return;
    onChange([...items, value]);
    setDraft("");
  };

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id}>{title}</Label>
      <ul className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <li key={item}>
            <span className="flex items-center gap-1 rounded-md border px-2 py-1 text-xs">
              {item}
              <button
                type="button"
                aria-label={`Remove ${item}`}
                disabled={disabled}
                onClick={() => onChange(items.filter((i) => i !== item))}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="size-3" />
              </button>
            </span>
          </li>
        ))}
        {items.length === 0 && (
          <li className="text-xs text-muted-foreground">{emptyHint}</li>
        )}
      </ul>
      <div className="flex gap-2">
        <Input
          id={id}
          value={draft}
          placeholder={placeholder}
          disabled={disabled}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              // The list often sits inside a form-shaped layout; Enter must add
              // a chip, never navigate away from a half-finished list.
              e.preventDefault();
              add();
            }
          }}
          className="max-w-xs"
        />
        <Button variant="outline" size="sm" disabled={disabled} onClick={add}>
          <Plus className="size-4" /> Add
        </Button>
      </div>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
