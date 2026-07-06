import type { LucideIcon } from "lucide-react";
import { Cpu, LineChart, Sigma } from "lucide-react";

export interface NavItem {
  label: string;
  to: string;
  icon: LucideIcon;
}

export const navItems: NavItem[] = [
  { label: "Playground", to: "/playground", icon: Cpu },
  { label: "Tensor Arithmetic", to: "/tensor", icon: Sigma },
  { label: "Training", to: "/training", icon: LineChart },
];
