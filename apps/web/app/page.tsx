import type { Metadata } from "next";
import { AutoscalerStudio } from "./AutoscalerStudio";

export const metadata: Metadata = {
  title: "Autoscaler · AI Theorist",
  description:
    "Compose a residual architecture, choose data and training budgets, validate transfer, and calibrate a held-out scaling law.",
  other: {
    "codex-preview": "development",
  },
};

export default function Home() {
  return <AutoscalerStudio />;
}
