import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

import { cn } from "@/lib/utils"

interface MarkdownProps {
  children: string
  className?: string
}

/**
 * Renders Markdown (GitHub-flavored) — intended for LLM/chat output.
 * Tailwind classes are applied per element so it works without the
 * `@tailwindcss/typography` plugin.
 */
export function Markdown({ children, className }: MarkdownProps) {
  return (
    <div
      className={cn(
        "space-y-3 text-sm leading-relaxed text-foreground",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ ...props }) => (
            <h1 className="text-xl font-semibold" {...props} />
          ),
          h2: ({ ...props }) => (
            <h2 className="text-lg font-semibold" {...props} />
          ),
          h3: ({ ...props }) => (
            <h3 className="text-base font-semibold" {...props} />
          ),
          ul: ({ ...props }) => (
            <ul className="list-disc space-y-1 pl-5" {...props} />
          ),
          ol: ({ ...props }) => (
            <ol className="list-decimal space-y-1 pl-5" {...props} />
          ),
          a: ({ ...props }) => (
            <a
              className="font-medium text-primary underline underline-offset-4"
              target="_blank"
              rel="noreferrer"
              {...props}
            />
          ),
          code: ({ className: codeClass, ...props }) => (
            <code
              className={cn(
                "rounded bg-muted px-1.5 py-0.5 font-mono text-[0.85em]",
                codeClass
              )}
              {...props}
            />
          ),
          pre: ({ ...props }) => (
            <pre
              className="overflow-x-auto rounded-lg bg-muted p-3 text-[0.85em]"
              {...props}
            />
          ),
          blockquote: ({ ...props }) => (
            <blockquote
              className="border-l-2 border-border pl-3 text-muted-foreground"
              {...props}
            />
          ),
          table: ({ ...props }) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left" {...props} />
            </div>
          ),
          th: ({ ...props }) => (
            <th className="border border-border px-2 py-1 font-medium" {...props} />
          ),
          td: ({ ...props }) => (
            <td className="border border-border px-2 py-1" {...props} />
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
