# GRPO and online RL, through 2048

A 4 minute 52 second narrated explainer for our first post-training experiment.
Includes 11 independently replayable chapters, measured phrase captions, real
simulator board transitions, a worked advantage example, and a clipping diagram.

## Watch and edit

Keep the chosen `REMOTION_PORT` in an ignored `.local/visuals.env` file at the
repository root. Local port values and credentials are not versioned.

```sh
cd grpo-explainer
npm ci
# On macOS, with ffmpeg installed:
npm run audio
. ../.local/visuals.env
export REMOTION_PORT
npm run dev
```

Open the printed Studio URL, select **GRPO-2048**, and press Play (or Space).
Open **Chapters** in the compositions menu to replay individual explanations.
The local preview already has narration. Speech assets are generated locally
with the Mac's Samantha voice and kept out of Git. Creating them needs macOS and
ffmpeg, with no model API, GPU, or API key. After generation, previewing uses the
local MP3 files.

| Start | Chapter                             |
| ----- | ----------------------------------- |
| 0:00  | How playing changes a model         |
| 0:24  | Board, policy, action, reward       |
| 0:48  | What makes learning online?         |
| 1:11  | Grouped rollouts                    |
| 1:36  | Reward design                       |
| 2:01  | Relative advantage                  |
| 2:30  | Policy update and credit assignment |
| 2:58  | Clipping and optional KL            |
| 3:25  | Failure modes                       |
| 3:51  | Our proposed TRL experiment         |
| 4:23  | Recap                               |

Read the full [transcript](TRANSCRIPT.md) or edit the original [script](narration.json).

## What is illustrative?

- Board animations use valid moves from this repository's simulator. The four
  demonstration trajectories use uniform random legal actions and independent
  environment seeds. They do not represent a trained language model.
- The four rewards `0, 1, 2, 3` are a separate teaching example. Centered rewards
  are `−1.5, −0.5, 0.5, 1.5`. The normalized illustration uses population standard
  deviation `sqrt(1.25)` and rounded advantages `−1.34, −0.45, 0.45, 1.34`.
  TRL supports other normalization choices; inspect the pinned implementation.
- The clipping diagram is a schematic for a positive advantage. Clipping changes
  the objective's incentive; it does not enforce a hard probability-ratio bound.
  A reference-policy KL term is optional and distinct from the rollout policy.
- Four trajectories, 16 attempts, the reward formula, and the 10-update inspection
  are the proposed experiment in [the project plan](../docs/PLAN.md). Neither SFT
  nor online RL had run when the video was authored. Subsequent one-update MPS
  compatibility checks are documented in [the local report](../docs/MPS_TRL_CHECK.md).
  No learning curves or improvement claims are shown.

## Regenerate and verify

```sh
# macOS with Samantha voice and ffmpeg on PATH; no paid service
npm run audio
npm run lint
npm run review
```

`prepare_assets.py` synthesizes each phrase with macOS `say`, measures its PCM
length, adds short gaps, normalizes narration, and writes the caption clips,
chapter durations, and transcript. Cached speech is ignored by Git. Regeneration
can vary slightly with the installed macOS voice version. The checked-in timing JSON records the narration timing used for this composition.
Regeneration updates the audio, captions, and composition durations together.

Review stills are written to ignored `out/review/`. The authored deliverable is
the interactive Studio composition; a full video file is not exported by default.

## Sources and further reading

- [Hugging Face TRL: GRPO Trainer](https://huggingface.co/docs/trl/en/grpo_trainer)
  — online collection, grouped advantages, configurable scaling, clipping and KL.
- [DeepSeekMath, original GRPO paper](https://arxiv.org/abs/2402.03300)
  — the group baseline and removal of a separately trained value critic.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  — clipped surrogate objectives and repeated updates using collected samples.
- [Our GPU handoff](../docs/GPU_HANDOFF.md) — policy actions, observation masks,
  independent random spawns, adapter checks, and full-game evaluation.

The original script and project code use the repository's MIT license.
System speech assets are separate from that source-code license. Third-party
packages retain their own licenses. `BasicCaptions.tsx` is the unmodified
[Remotion Basic Captions element](https://www.remotion.dev/elements/captions/basic-captions).
Remotion's software terms apply separately to use of its tools.
