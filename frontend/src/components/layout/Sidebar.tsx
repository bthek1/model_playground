import { useEffect } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { ChevronRight, Home } from "lucide-react";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/ui";
import { categoryForPath, taskCategories } from "./taskTaxonomy";

export function SidebarNav({
  collapsed = false,
  onNavigate,
}: {
  collapsed?: boolean;
  onNavigate?: () => void;
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const expandedCategories = useUIStore((s) => s.expandedCategories);
  const toggleCategory = useUIStore((s) => s.toggleCategory);
  const setCategoryExpanded = useUIStore((s) => s.setCategoryExpanded);
  const setSidebarOpen = useUIStore((s) => s.setSidebarOpen);

  // Auto-expand the category that owns the active route.
  const activeCategory = categoryForPath(pathname);
  useEffect(() => {
    if (activeCategory) setCategoryExpanded(activeCategory, true);
  }, [activeCategory, setCategoryExpanded]);

  const isHomeActive = pathname === "/home";

  return (
    <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-2">
      {/* Top-level Home link, above the task taxonomy. */}
      <Link
        to="/home"
        onClick={onNavigate}
        aria-current={isHomeActive ? "page" : undefined}
        title={collapsed ? "Home" : undefined}
        className={cn(
          "flex items-center rounded-md text-sm transition-colors",
          isHomeActive
            ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
            : "text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
          collapsed ? "justify-center p-2" : "gap-3 px-3 py-2",
        )}
      >
        <Home className="h-5 w-5 shrink-0" />
        {!collapsed && <span>Home</span>}
      </Link>

      {taskCategories.map((category) => {
        const isExpanded = expandedCategories[category.label] ?? false;

        // Collapsed rail: one icon per category. Clicking expands the rail and
        // opens that category.
        if (collapsed) {
          return (
            <button
              key={category.label}
              type="button"
              title={category.label}
              onClick={() => {
                setSidebarOpen(true);
                setCategoryExpanded(category.label, true);
              }}
              className="flex items-center justify-center rounded-md p-2 text-sidebar-foreground transition-colors hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground"
            >
              <category.icon className="h-5 w-5 shrink-0" />
            </button>
          );
        }

        return (
          <div key={category.label} className="flex flex-col">
            <button
              type="button"
              aria-expanded={isExpanded}
              onClick={() => toggleCategory(category.label)}
              className="flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-sidebar-foreground transition-colors hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground"
            >
              <category.icon className="h-5 w-5 shrink-0" />
              <span className="flex-1 text-left">{category.label}</span>
              <ChevronRight
                className={cn(
                  "h-4 w-4 shrink-0 transition-transform",
                  isExpanded && "rotate-90",
                )}
              />
            </button>

            {isExpanded && (
              <ul className="mt-1 flex flex-col gap-0.5 pl-4">
                {category.tasks.map((taskItem) => {
                  const isActive = pathname === taskItem.to;
                  return (
                    <li key={taskItem.slug}>
                      <Link
                        to={taskItem.to}
                        onClick={onNavigate}
                        aria-current={isActive ? "page" : undefined}
                        className={cn(
                          "flex items-center rounded-md px-3 py-1.5 text-sm transition-colors",
                          isActive
                            ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                            : "text-sidebar-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                        )}
                      >
                        <span className="truncate">{taskItem.label}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        );
      })}
    </nav>
  );
}

export function Sidebar() {
  const sidebarOpen = useUIStore((s) => s.sidebarOpen);

  return (
    <aside
      className={cn(
        "hidden md:flex flex-col border-r bg-sidebar shrink-0 transition-[width] duration-300 ease-in-out overflow-hidden",
        sidebarOpen ? "w-64" : "w-16",
      )}
    >
      {/* Logo */}
      <div
        className={cn(
          "flex h-14 items-center border-b shrink-0",
          sidebarOpen ? "px-4" : "justify-center",
        )}
      >
        <span className="font-semibold text-sidebar-foreground truncate">
          {sidebarOpen ? "Model Playground" : "MP"}
        </span>
      </div>

      <SidebarNav collapsed={!sidebarOpen} />
    </aside>
  );
}
