# ESP32-S3 6.34M LSTM usefulness probe — 2026-07-01

Project: `/home/sikmindz/projects/esp32-s3-lstm-proof`

## Bottom line

Yes: the raw achievement is phenomenal for MCU-class hardware.

But the current model should not be used as a free-form decision brain yet. The best use of the compute we have right now is:

1. deterministic local sentinel / policy decision
2. receipt emission
3. LSTM-generated short status/explanation flavor, constrained to 16-48 chars
4. escalation to UNO Q / GTX / Gemma when the deterministic sentinel says the situation needs real language or richer reasoning

That makes the ESP32-S3 useful now without pretending the 6.34M char-LSTM is a general assistant.

## What I built to test this

Added host-side usefulness probe:

`/home/sikmindz/projects/esp32-s3-lstm-proof/tools/usefulness_probe.py`

It parses the actual deployed RILM v1 weight file:

`/home/sikmindz/projects/esp32-s3-lstm-proof/weights.bin`

It mirrors the LSTM forward pass, then tests 21 useful prompts across greedy and top-k sampling modes.

Receipt:

`/home/sikmindz/projects/esp32-s3-lstm-proof/analysis/usefulness_probe_receipt.json`

Command run:

```bash
cd /home/sikmindz/projects/esp32-s3-lstm-proof
python3 tools/usefulness_probe.py --length 96
```

Observed output summary:

```json
{
  "best_config": {
    "mode": "sample",
    "temperature": 0.8,
    "top_k": 8,
    "mean_score": 36.35,
    "best_score": 57.38,
    "worst_score": 14.79
  },
  "hardware_budget": {
    "16_chars_ms": 4340.3,
    "32_chars_ms": 8680.6,
    "64_chars_ms": 17361.1,
    "96_chars_ms": 26041.7
  }
}
```

Important: the scoring is only a utility screen, not a scientific quality metric. It checks domain terms, repetition, degenerate fragments, and rough text shape. The raw samples matter more than the score.

## Best current behavior

Some prompts do activate the edge-AI training mixture.

Best useful samples from the actual current weights:

Prompt:
`normal room. action is `

Generated:
`issing. the gateway writes a receipt before it takes action. the sensor sees warm air and heavy `

Prompt:
`do not invent `

Generated:
`eadings. do not pretend the room is safe if the sensor is missing. the gateway decides what matt`

Prompt:
`if data is stale `

Generated:
`he model must say the data is stale. the sensor sees warm air and heavy humidity. the gateway de`

This is the useful signal: the model learned some of the local-first / sensor / receipt / stale-data language. It can emit short, on-theme status text.

## Failure mode

It also falls back into TinyStories-style narrative mush.

Bad samples:

Prompt:
`hot room. action is `

Generated:
`o scared. she said, yes, it's a small spot. it's a small spot. it's a small spot. it's a small s`

Prompt:
`humid room. action is `

Generated:
`o excited.\nthe little boy was so excited. she said, it's mine, said the boy. you are so much bet`

That means current weights are not suitable for autonomous policy or safety-critical text. The model has enough capacity and speed, but the training distribution is still wrong for the product use.

## Compute budget reality

Hardware reference from p7:

- 3.6864 tok/s
- 271.27 ms/token
- 6,337,569 params
- 4,785,276-byte packed model

At this speed:

| Output length | Latency |
|---:|---:|
| 16 chars | 4.34 s |
| 32 chars | 8.68 s |
| 48 chars | 13.02 s |
| 64 chars | 17.36 s |
| 96 chars | 26.04 s |

So the product target is not paragraph generation. It is short local status bursts.

Good UX target:

- instant deterministic decision: <100 ms
- local LSTM status: 16-32 chars, 4-9 s
- richer explanation: escalate to bigger model

## Most useful architecture with current compute

Use the ESP32-S3 as a local physical sentinel.

Pipeline:

1. Read sensors.
2. Compute deterministic local state:
   - normal
   - warm
   - humid
   - stale
   - missing sensor
   - escalate
3. Emit a receipt immediately.
4. Display/send a short deterministic status string.
5. Optionally let the LSTM generate a short 16-32 char local phrase.
6. Escalate to Gemma/Ollama only for rich explanation, multi-step advice, or ambiguous state.

The LSTM should decorate or compress a decision. It should not decide.

## What to change next for actual usefulness

### 1. Retrain/domain-distill the same architecture on receipt/action text

This is the highest ROI.

Current model is mostly TinyStories with a small edge-AI mixture. That is why it sometimes says useful things, then falls into children-story text.

New corpus should be mostly:

- sensor receipt lines
- status summaries
- short action labels
- stale/missing-data warnings
- Home Assistant style entity text
- local-first physical AI phrases
- safety negatives: “do not invent readings”

Target output examples:

- `room normal. receipt logged.`
- `humid air. fan advised.`
- `sensor stale. wait for data.`
- `heat rising. escalate.`
- `missing dht. no claim.`

This may make an H256/H384 model more useful than the current H512 model.

### 2. Constrained decoding, not free-form generation

Current greedy repeats. Sampling helps but can drift.

Firmware should support:

- top-k sampling over 33 chars
- temperature around 0.45-0.8
- stop at newline / period / max chars
- ban obvious repetition loops
- optional prefix templates

Best host-side mean score in this probe was top-k sampling, not greedy.

### 3. H384/H256 domain model

The useful target probably does not need 6.34M params.

Expected trade:

- H512 current: 3.69 tok/s, mediocre domain reliability
- H384 domain model: likely faster and possibly more useful
- H256 domain model: maybe much faster, likely good enough for status/receipt language

The real question is not “largest model on ESP32.” It is “smallest model that reliably emits local physical status text.”

### 4. Keep p7 speed work, but do not over-optimize generic text yet

The next speed work is still valid:

- int8/ESP-NN path for `lstm_wih`
- RILM v2 aligned layout
- H384/H256 export

But the bigger product win is data/decoding. Speed matters less if the model is generating the wrong kind of text.

## Safe claim

Safe:

“RecursiveIntell has a 6.34M-param compressed char-LSTM running locally on ESP32-S3 at 3.69 tok/s. A usefulness probe shows the current model has learned some sensor/receipt/local-first language, but is not reliable as a free-form decision model. The immediate product path is deterministic sentinel + receipt + short constrained LSTM status text, with escalation for richer reasoning.”

Do not claim:

- it is a useful standalone chatbot
- it can make safe physical decisions by itself
- it reliably understands sensor context
- it beats other LMs on quality

## Verdict

Keep.

This is not just a stunt. The compute is enough for a useful local endpoint if we stop asking it to be a chatbot and make it a constrained physical-status generator.

The highest ROI next work is not another micro-optimization pass. It is:

1. train/domain-distill H512/H384/H256 on receipt/action/status text
2. export mixed precision
3. run this usefulness probe + PPL
4. flash best candidate
5. measure hardware tok/s and actual status samples
