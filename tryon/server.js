// Minimal Try-on HTTP service stub for Runpod Dedicated Pod
// Exposes POST /tryon { lookUrl, onYouUrl } -> { tryonUrl, items[] }
// This version performs a simple overlay of the look image on the on-you image for demo purposes.

import express from 'express';
import fetch from 'node-fetch';
import sharp from 'sharp';

const app = express();
app.use(express.json({ limit: '20mb' }));

async function fetchImageBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

app.get('/health', (_req, res) => res.json({ ok: true }));

app.post('/tryon', async (req, res) => {
  try {
    const { lookUrl, onYouUrl } = req.body;
    if (!lookUrl || !onYouUrl) return res.status(400).json({ error: 'lookUrl and onYouUrl required' });
    const [lookBuf, baseBuf] = await Promise.all([fetchImageBuffer(lookUrl), fetchImageBuffer(onYouUrl)]);
    const meta = await sharp(baseBuf).metadata();
    const W = meta.width || 1080; const H = meta.height || 1920;
    const lookW = Math.round(W * 0.35);
    const look = await sharp(lookBuf).resize({ width: lookW, fit: 'inside' }).png().toBuffer();
    const composed = await sharp(baseBuf)
      .composite([{ input: look, left: Math.round(W * 0.06), top: Math.round(H * 0.06), blend: 'over', opacity: 0.9 }])
      .png({ compressionLevel: 9 }).toBuffer();
    // Return as data URL for demo; in production, upload to object storage and return a URL
    const b64 = composed.toString('base64');
    res.json({ tryonUrl: `data:image/png;base64,${b64}` , items: [ { n:1, label:'Top' }, { n:2, label:'Bottom' } ] });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Tryon stub listening on :${port}`));

