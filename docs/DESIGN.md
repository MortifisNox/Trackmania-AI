# Design

## Constraints that drive every choice

- **Trackmania 2020 runs in real time only.** It has no save states and no
  speed-up we can legitimately use (TMInterface, which gives both, targets
  Trackmania Nations Forever; speed changes in TM2020 are the classic cheat,
  see SAFE_USE.md). One laptop runs one game instance, so the agent gets about
  **20 decisions per second, ~72k transitions per hour**. Sample efficiency is
  the deciding factor.
- **One 6 GB GPU shared with the game.** The game needs roughly 1.5 to 2.5 GB
  at low settings *(estimate, to be measured in phase 1)*; the learner gets a
  hard cap of 50% of VRAM.
- **Laptop thermals.** Multi-day runs must survive throttling without
  corrupting anything.
- **Continuous control and surface changes.** Analog steering/throttle, and
  grip changes on dirt, grass, ice, plastic and water.

## Methodology: options considered

| Approach | Fit for TM2020 on one laptop | Verdict |
|---|---|---|
| **SAC / REDQ (TMRL-style)**: off-policy actor-critic, replay buffer, continuous actions | Reuses every transition many times, handles analog control, proven on TM2020 by TMRL ([repo](https://github.com/trackmania-rl/tmrl)) | **Base of our design** |
| PPO (on-policy) | Throws data away after each update; needs far more samples than real time gives | Rejected |
| IQN / DQN with discrete actions (Linesight-style) | Superhuman results on TMNF, but those rely on many parallel game instances with game speed-up via TMInterface *(from memory, not re-verified)*, which TM2020 does not allow | Rejected as the core; its ideas (distributional critics) remain options |
| Neuroevolution (NEAT, CMA-ES) | Needs thousands of full-episode rollouts per generation; at real-time speed that is weeks per improvement | Rejected |
| Pure imitation of human laps | Fast start, but caps at the demonstrator's skill | Used only to **seed** RL |

## Chosen approach

**SAC with a DroQ critic ensemble, seeded with human demonstrations, and a
self-improving reference line.**

1. **SAC + DroQ.** DroQ (Hiraoka et al., 2021) reaches REDQ-level sample
   efficiency with 2 critics that use dropout and LayerNorm instead of REDQ's
   10, which matters on 6 GB and lets us run a high update-to-data ratio
   (default 4 updates per environment step). TMRL ships REDQ, which confirms
   high-UTD SAC variants work in this game.
2. **Demonstration seeding (RLPD-style).** You drive a few laps of each new
   track; every training batch draws 50% from those laps and 50% from the
   agent's own experience. This removes the long random-driving phase.
3. **Observations.** 4 stacked 64x64 grayscale frames (sees surface types and
   walls) plus a state vector from the Openplanet plugin: speed, gear, RPM,
   wheel contact and surface material, velocity in the car frame, previous
   action, and 10 points of the reference line ahead of the car. Random-shift
   image augmentation (DrQ) against overfitting.
4. **Reward.** Progress along the reference line per step (as TMRL does), a
   penalty for wall contact and a time penalty, plus a bonus for beating the
   best checkpoint split. Episodes end on finish, being stuck, or timeout.
5. **Continuous improvement.** The reference line starts as your demo lap.
   Whenever the agent sets a new best lap, that lap becomes the reference, so
   progress reward and lookahead follow ever-faster lines instead of anchoring
   to the human one. Evaluation laps run with the deterministic policy.
6. **Arbitrary tracks.** Inputs are local (camera frames, car-frame
   lookahead), not track coordinates, so one policy can train across a set of
   tracks and be fine-tuned per track.

## Architecture

```
 Trackmania + Openplanet plugin ──UDP telemetry (localhost)──┐
        ▲                                                    ▼
 virtual Xbox pad (vgamepad) ◀── actor process (20 Hz loop, policy inference,
        ▲                         screen capture with dxcam, episode logic)
        │                                   │ transitions        ▲ weights
        │                                   ▼                    │
        │                         learner process (GPU, SAC/DroQ, replay in RAM)
        │                                   │ metrics            ▲ commands
        │                                   ▼                    │
        └── manual reset ◀──────── dashboard (FastAPI + browser, localhost:8080)
                                   + TensorBoard
```

- **Two processes** so a slow gradient step never delays a control step.
- **Replay buffer in system RAM** as uint8 frames stored once (stacks rebuilt
  on sampling): ~4.2 KB per transition, 500k transitions ≈ 2 GB. `tmai doctor`
  suggests a capacity for your RAM.
- **VRAM budget**: ~2M-parameter networks, batch 256, bf16 autocast on the
  4050's tensor cores with fp32 master weights. Expected under 1.5 GB; capped
  at 50% with `torch.cuda.set_per_process_memory_fraction`.
- **Thermal governor**: NVML temperature polling; above 80 °C the UTD ratio is
  halved, above 87 °C learning pauses (driving continues) until it cools.
- **Crash-safe state**: checkpoints and replay snapshots are written to a temp
  file and atomically renamed, so a power cut or crash leaves the previous
  good copy. Shutdown (dashboard button, Ctrl+C or SIGTERM) stops the actor,
  drains the queue, writes the final checkpoint and replay snapshot, then
  builds the run report.

## Dashboard

Local web page served by the learner (FastAPI + WebSocket), plus TensorBoard
for history:

- Live: speed, inputs, current lap time, checkpoint splits against the best
  lap, episode reward, losses, entropy, Q-values, GPU temperature/VRAM, UTD.
- Controls: pause/resume, change learning rate / UTD / entropy target /
  exploration at runtime, force a track reset, switch track, save checkpoint,
  stop safely.

## Run report and best ghost

On stop: best lap and splits per track, split comparison against your demo
lap, total steps and hours, reward/loss/Q curves with a stability summary
(rolling variance), exported as HTML in `runs/<name>/report/`. The best lap is
kept as a game replay: Trackmania autosaves replays of personal bests in local
play, and the report links it for viewing in the replay editor *(the exact
mechanism, autosave folder versus a plugin call, gets verified in phase 6)*.
