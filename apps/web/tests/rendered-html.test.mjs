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
  assert.match(html, /Choose the regime\. Generate the ladder/);
  assert.match(html, /Forecasts must earn the right to appear/);
  assert.doesNotMatch(html, /Your site is taking shape/);
});

test("includes the held-out validation contract", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /Validation examples always remain fixed/);
  assert.match(html, /common validation/i);
  assert.match(html, /largest held out/i);
  assert.match(html, /Your held-out result lands here/);
});

test("offers the nuGPT normalized-Transformer transfer contract", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /νGPT/);
  assert.match(html, /Normalized Transformer with LR transfer/);
});

test("defaults residual branches to inverse-depth scaling", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /Branch multiplier/);
  assert.match(html, /1 \/ R/);
  assert.doesNotMatch(html, /1 \/ √R/);
});

test("exposes profiles, datasets, generated ladders, and joint scaling", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /Plumbing only · forecasts disabled/);
  assert.match(html, /Find transfer and loss range/);
  assert.match(html, /Teacher regression/);
  assert.match(html, /Markov language/);
  assert.match(html, /Automatic scale ladder/);
  assert.match(html, /Token budget|Sample budget/);
  assert.match(html, /Joint model \+ data/);
  assert.match(html, /power-law readiness/i);
});
