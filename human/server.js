// Minimal Human HTTP service for Runpod Dedicated Pod
// Exposes:
//   POST /pose { imageUrl } -> { keypoints:[{x,y,score}], width, height }
//   POST /segment { imageUrl } -> { maskUrl, width, height }

import express from 'express';
import fetch from 'node-fetch';
import Human from '@vladmandic/human';
import sharp from 'sharp';

const app = express();
app.use(express.json({ limit: '20mb' }));

const human = new Human({
  modelBasePath: 'https://vladmandic.github.io/human/models',
  face: { enabled: false },
  hand: { enabled: false },
  body: { enabled: true },
  segmentation: { enabled: true },
});

async function fetchImageBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

app.get('/health', async (_req, res) => {
  res.json({ ok: true });
});

app.post('/pose', async (req, res) => {
  try {
    const { imageUrl } = req.body;
    const buf = await fetchImageBuffer(imageUrl);
    const { data, info } = await sharp(buf).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    await human.load();
    const tensor = human.tf.tensor(data, [info.height, info.width, 4], 'int32');
    const result = await human.detect(tensor);
    human.tf.dispose(tensor);
    const kp = (result.body?.[0]?.keypoints || []).map(k => ({ x: k.position.x / info.width, y: k.position.y / info.height, score: k.score }));
    res.json({ keypoints: kp, width: info.width, height: info.height });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.post('/segment', async (req, res) => {
  try {
    const { imageUrl } = req.body;
    const buf = await fetchImageBuffer(imageUrl);
    const img = await sharp(buf).ensureAlpha().resize({ width: 640, fit: 'inside' }).toBuffer();
    const { data, info } = await sharp(img).raw().toBuffer({ resolveWithObject: true });
    await human.load();
    const tensor = human.tf.tensor(data, [info.height, info.width, 4], 'int32');
    const result = await human.detect(tensor);
    human.tf.dispose(tensor);
    const mask = result.segmentation?.data?.person || result.segmentation?.mask?.data; // [h*w] 0..1
    if (!mask) return res.json({ maskUrl: imageUrl, width: info.width, height: info.height });
    const alpha = Buffer.alloc(info.width * info.height);
    for (let i = 0; i < alpha.length; i++) alpha[i] = Math.round(255 * mask[i]);
    const out = await sharp(img).ensureAlpha().joinChannel(alpha).png().toBuffer();
    const b64 = out.toString('base64');
    res.json({ maskUrl: `data:image/png;base64,${b64}`, width: info.width, height: info.height });
  } catch (e) {
    res.status(500).json({ error: String(e.message || e) });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Human service listening on :${port}`));

