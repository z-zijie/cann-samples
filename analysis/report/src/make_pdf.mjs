// Render a print-optimized HTML file to a professional multi-page PDF via Playwright + Chromium.
// Usage: node make_pdf.mjs <input.html> <output.pdf>
import { chromium } from 'playwright';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [, , inFile, outFile] = process.argv;
if (!inFile || !outFile) {
  console.error('Usage: node make_pdf.mjs <input.html> <output.pdf>');
  process.exit(1);
}

const url = pathToFileURL(path.resolve(inFile)).href;

const browser = await chromium.launch({
  executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
  args: ['--no-sandbox', '--disable-gpu', '--font-render-hinting=none'],
});
const page = await browser.newPage();
await page.emulateMedia({ media: 'print' });
await page.goto(url, { waitUntil: 'networkidle' });
// Give inline SVG / layout a beat to settle.
await page.waitForTimeout(400);

await page.pdf({
  path: outFile,
  format: 'A4',
  printBackground: true,
  preferCSSPageSize: true,
  displayHeaderFooter: true,
  margin: { top: '14mm', bottom: '16mm', left: '0mm', right: '0mm' },
  headerTemplate: '<div></div>',
  footerTemplate:
    '<div style="width:100%;font-family:\'Liberation Sans\',sans-serif;font-size:7.5px;color:#8a8f98;padding:0 14mm;display:flex;justify-content:space-between;align-items:center;">' +
    '<span>cann-samples  vs  cuda-samples · 开源生态深度对比报告</span>' +
    '<span>第 <span class="pageNumber"></span> / <span class="totalPages"></span> 页</span>' +
    '</div>',
});

await browser.close();
console.log('PDF written to', outFile);
