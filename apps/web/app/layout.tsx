import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Autoscaler · AI Theorist",
  description: "Explicit-budget neural scaling with honest held-out calibration.",
  openGraph: {
    title: "Autoscaler · AI Theorist",
    description: "Compose, tune, transfer, calibrate, and forecast neural architectures.",
    images: [{ url: "/autoscaler-social-preview.png", width: 1536, height: 1024 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Autoscaler · AI Theorist",
    description: "Explicit-budget neural scaling with honest held-out calibration.",
    images: ["/autoscaler-social-preview.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
