import { bundle } from "@remotion/bundler";
import { getCompositions, openBrowser, renderStill } from "@remotion/renderer";
import { mkdir, readFile } from "node:fs/promises";

// Render representative teaching frames, not a full video export.
const timing = JSON.parse(await readFile("public/timing.json", "utf8"));
await mkdir("out/review", { recursive: true });
const serveUrl = await bundle({ entryPoint: "src/index.ts" });
const browser = await openBrowser("chrome");
try {
  const compositions = await getCompositions(serveUrl, {
    puppeteerInstance: browser,
  });
  for (const scene of timing.scenes) {
    const composition = compositions.find((item) => item.id === scene.id);
    const frame = Math.min(scene.frames - 30, 600);
    await renderStill({
      serveUrl,
      composition,
      frame,
      output: `out/review/${scene.id}.png`,
      puppeteerInstance: browser,
      imageFormat: "png",
    });
    console.log(`${scene.id}: frame ${frame}`);
  }
} finally {
  await browser.close({ silent: true });
}
