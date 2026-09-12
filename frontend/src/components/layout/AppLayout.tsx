import { Outlet } from "@tanstack/react-router";
import { Navbar } from "./Navbar";
import { RightPanel } from "./RightPanel";
import { Sidebar } from "./Sidebar";

export function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <Navbar />
        {/* The system panel sits beside `main`, not inside it: a route's own
            grid is laid out on viewport breakpoints and must keep the element
            it measures against. `min-w-0` is what lets main give way. */}
        <div className="flex min-h-0 flex-1">
          <main className="min-w-0 flex-1 overflow-auto p-6">
            <Outlet />
          </main>
          <RightPanel />
        </div>
      </div>
    </div>
  );
}
