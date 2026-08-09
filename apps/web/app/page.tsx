import type { Metadata } from "next";
import { AutoscalerStudio } from "./AutoscalerStudio";

export const metadata: Metadata = {
  title: "Autoscaler · AI Theorist",
  description:
    "Compose a residual architecture, tune it, validate transfer, and calibrate a fixed-horizon scaling law.",
  other: {
    "codex-preview": "development",
  },
};

export default function Home() {
  return <AutoscalerStudio />;
}
