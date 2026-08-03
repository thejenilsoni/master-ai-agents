# Image Q&A Agent (Multimodal)

A **beginner** introduction to giving a chat model eyes. Point it at one or more
local image files, ask a question, and get an answer grounded in what is
actually in the pixels — a whiteboard sketch, a shelf of price tags, a
photograph.

The interesting part is not the API call, it is everything around it: turning a
file on disk into a `data:` URL, deciding how many pixels are worth sending, and
knowing what a picture costs before you send it. Those are all plain functions
here, so you can inspect and test them without spending anything.

No image files are committed to this repository. `make_samples.py` draws the
sample images on your machine with Pillow.

## What it demonstrates

- **Base64 data URLs** — `encode_image_data_url()` reads a file and produces
  `data:image/png;base64,…`, and `decode_data_url()` inverts it so the encoding
  path is verifiable offline.
- **Several images in one request** — the user message `content` is a list: one
  text part followed by one `image_url` part per image, numbered in the prompt
  so the model can refer to them.
- **The `detail` setting, priced** — `estimate_image_tokens()` reproduces the
  published tiling arithmetic: `low` is a flat base cost, `high` shrinks the
  image to a 512-px tile grid and charges per tile.
- **Why downscaling matters — and where it doesn't.** High detail normalises
  every image to the same tile grid, so a 4000x3000 photo and a 1024x768 one
  cost *identical* tokens. Downscaling to 1024 still cuts upload bytes and
  latency by an order of magnitude; only pushing the short side below 768 px
  actually reduces the token bill, and that is exactly the point where small
  text stops being readable.
- **Guardrails** — an unsupported file type, a payload over the ~20 MB limit, or
  more than four images per request all fail early with a clear message.

## How to Get Started

### 1. Clone and enter the project

```bash
git clone https://github.com/thejenilsoni/master-ai-agents.git
cd master-ai-agents/multimodal/beginner/image-qa-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the sample images

```bash
python make_samples.py
```

This writes three files into `samples/` (git-ignored):

| File | Size | What it is for |
| --- | --- | --- |
| `whiteboard.png` | 1280x900 | An architecture sketch — boxes, arrows, a notes box. |
| `shelf_tags.png` | 1100x760 | Product tags with deliberately tiny unit-price print. |
| `trail_photo.jpg` | 2400x1800 | A large "photo" with a small signpost caption. |

### 4. Set your OpenAI API key

```bash
cp .env.example .env   # then edit .env
```

### 5. Run

```bash
# One image:
python image_qa_agent.py --image samples/whiteboard.png "What are the two open questions?"

# Two images in one request:
python image_qa_agent.py \
  --image samples/whiteboard.png --image samples/shelf_tags.png \
  "What do these two images have in common, and how do they differ?"

# Read the small print — high detail is worth paying for here:
python image_qa_agent.py --image samples/shelf_tags.png --detail high \
  "List every product with its price and its per-100 g unit price."

# Price a request without sending it:
python image_qa_agent.py --image samples/trail_photo.jpg --detail high --dry-run "x"
```

Useful flags: `--model gpt-4o` (stronger, pricier), `--max-side 512` (shrink
harder), `--detail low|high|auto`.

## Verify it without an API key

Encoding, downscaling, and cost estimation are pure functions with a built-in
self-test — no key, and not even Pillow, required:

```bash
python image_qa_agent.py --selftest
# selftest passed: data-URL round-trip, downscale math, tile/token estimates,
#   high-detail 4000x3000 == 1024x768 == 765 tokens, 512x384 == 255 tokens (gpt-4o)
```

## Example output

```
$ python image_qa_agent.py --image samples/trail_photo.jpg --detail high \
    "What does the signpost say?"

Prepared images:
  1. trail_photo.jpg              1024x768     37.1 KB  ~25501 tokens (high)
  estimated image tokens for gpt-4o-mini: ~25501

Q: What does the signpost say?

The signpost on the trail reads "Fern Hollow Trail - 3.2 km", with a smaller
line underneath giving an elevation gain of 240 m.
```

Now run the same command with `--max-side 384`. The image drops to one tile and
about a third of the tokens — and the model can no longer read the sign, which
is the trade-off made visible.

## A note on honesty

Vision models are confident readers of things that are not there. The system
prompt in this project explicitly asks the model to say when text is too small
or blurry to read, and the samples are built so you can catch it out: the unit
prices on `shelf_tags.png` are 15 px tall on purpose. Compare `--detail low` and
`--detail high` answers on that image before you trust either one in production.

## Extending this project

- Accept a URL instead of a local path — the API takes a plain `https://` image
  URL in the same `image_url` field, no base64 needed.
- Add `--crop x,y,w,h` so you can send just the region of interest at high
  detail instead of the whole frame.
- Cache the prepared data URL next to the source file so repeated questions
  about the same image skip the resize.
- Return structured output (see the receipt extractor in this category) instead
  of prose, so downstream code can consume the answer.
