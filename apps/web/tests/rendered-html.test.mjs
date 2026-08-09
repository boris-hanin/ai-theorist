import assert from "node:assert/strict";
import test from "node:test";

const developmentPreviewMeta =
  /<meta(?=[^>]*\bname=["']codex-preview["'])(?=[^>]*\bcontent=["']development["'])[^>]*>/i;

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Autoscaler product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, developmentPreviewMeta);
  assert.match(html, /<title>Autoscaler · AI Theorist<\/title>/i);
  assert.match(html, /Build the model/);
  assert.match(html, /Residual stack/);
  assert.match(html, /Chizat particles/);
  assert.match(html, /Muon/);
  assert.match(html, /U\/W Muon/);
  assert.match(html, /Trained/);
  assert.match(html, /One horizon, five scales/);
  assert.match(html, /Forecasts must earn the right to appear/);
  assert.doesNotMatch(html, /Your site is taking shape/);
});

test("includes the fixed-horizon validation contract", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /Fixed across every scale/);
  assert.match(html, /Common random seeds/);
  assert.match(html, /Largest level held out/);
  assert.match(html, /Your held-out result lands here/);
  assert.match(html, /Dataset task/);
});
