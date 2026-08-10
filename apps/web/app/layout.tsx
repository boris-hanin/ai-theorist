import type { Metadata } from "next";
import "./globals.css";

const siteOrigin = process.env.NEXT_PUBLIC_SITE_URL ?? "https://ai-theorist-autoscaler.bhanin.chatgpt.site";
const socialPreview = "/autoscaler-social-preview-v2.png";

export const metadata: Metadata = {
  metadataBase: new URL(siteOrigin),
  title: "Autoscaler · AI Theorist",
  description: "Explicit-budget neural scaling, real-text batch campaigns, and honest held-out calibration.",
  openGraph: {
    title: "Autoscaler · AI Theorist",
    description: "Compose models and run resumable real-text batch-scaling campaigns from FP32 to BF16 FlashAttention and FSDP.",
    images: [{ url: socialPreview, width: 1672, height: 941 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Autoscaler · AI Theorist",
    description: "Real-text neural scaling campaigns with inspectable transfer and held-out calibration.",
    images: [socialPreview],
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
