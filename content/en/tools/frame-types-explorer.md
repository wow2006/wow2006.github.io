---
title: "Frame Types Explorer"
date: 2026-08-03
draft: false
description: "Interactive explorer for I, P and B video frames, GOP structure, and decode/display reordering"
---

<script src="/tools/frame-types-explorer/support.js"></script>
<x-dc>
<helmet>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
<style>
  .frame-explorer * { box-sizing: border-box; }
</style>
</helmet>
<div class="frame-explorer" style="background: #08090a; color: #d6dcd9; font-family: Helvetica Neue, Helvetica, Arial, sans-serif; font-size: 15px; line-height: 1.6; padding: 32px; border-radius: 6px">

  <div style="max-width: 1176px; margin: 0 auto">

    <header style="padding: 56px 0 32px 0; border-bottom: 1px solid #1c2321; display: flex; gap: 40px; align-items: flex-end; justify-content: space-between; flex-wrap: wrap">
      <div style="max-width: 660px">
        <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: #62e0a1; margin-bottom: 18px">Video Encoding · Bench Notes 01</div>
        <h1 style="font-size: 46px; line-height: 1.08; margin: 0 0 18px 0; font-weight: 600; letter-spacing: -0.02em; color: #f0f4f2">I, P and B frames</h1>
        <p style="margin: 0; color: #9aa5a1; font-size: 17px; text-wrap: pretty">A compressed video is not a stack of pictures. It is one picture plus a long list of <em style="color: #d6dcd9; font-style: normal">instructions for changing it</em>. Everything else — GOP size, B-frame count, seek behaviour, latency, why a dropped packet smears green across your screen — falls out of that one idea. Turn the knobs below and watch it happen.</p>
      </div>
      <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #55605c; line-height: 1.9; text-align: right">
        <div>codec-agnostic model</div>
        <div>{{ fpsLabel }} · {{ totalLabel }} frames</div>
        <div>gstreamer mappings included</div>
      </div>
    </header>

    <section style="padding: 44px 0 8px 0">
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px">

        <div style="border: 1px solid #223028; border-top: 2px solid #62e0a1; background: linear-gradient(180deg, rgba(98,224,161,0.05), rgba(98,224,161,0)); padding: 22px 22px 24px 22px">
          <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px">
            <span style="font-family: IBM Plex Mono, monospace; font-size: 30px; font-weight: 600; color: #62e0a1; line-height: 1">I</span>
            <span style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #7f8b86">intra · keyframe</span>
          </div>
          <p style="margin: 0 0 12px 0; font-size: 14px; color: #b9c3bf">Coded entirely from itself. No other frame is consulted, so it is the only frame you can hand a decoder cold.</p>
          <p style="margin: 0; font-size: 13px; color: #7f8b86"><strong style="color: #9aa5a1; font-weight: 500">Analogy:</strong> the full JPEG. Expensive to store, but it is the door into the stream.</p>
          <div style="margin-top: 16px; padding-top: 14px; border-top: 1px dashed #223028; font-family: IBM Plex Mono, monospace; font-size: 11.5px; color: #62e0a1">≈ {{ statI }} · {{ shareI }} of all bits</div>
        </div>

        <div style="border: 1px solid #22303a; border-top: 2px solid #7fb6d6; background: linear-gradient(180deg, rgba(127,182,214,0.05), rgba(127,182,214,0)); padding: 22px 22px 24px 22px">
          <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px">
            <span style="font-family: IBM Plex Mono, monospace; font-size: 30px; font-weight: 600; color: #7fb6d6; line-height: 1">P</span>
            <span style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #7f8b86">predicted · forward</span>
          </div>
          <p style="margin: 0 0 12px 0; font-size: 14px; color: #b9c3bf">Stores only what changed since earlier frames: block motion vectors plus a small residual for what motion could not explain.</p>
          <p style="margin: 0; font-size: 13px; color: #7f8b86"><strong style="color: #9aa5a1; font-weight: 500">Analogy:</strong> a diff against the last commit. Useless without the commit it was diffed against.</p>
          <div style="margin-top: 16px; padding-top: 14px; border-top: 1px dashed #22303a; font-family: IBM Plex Mono, monospace; font-size: 11.5px; color: #7fb6d6">≈ {{ statP }} · {{ shareP }} of all bits</div>
        </div>

        <div style="border: 1px solid #2a2c2b; border-top: 2px solid #8d9a94; background: linear-gradient(180deg, rgba(141,154,148,0.05), rgba(141,154,148,0)); padding: 22px 22px 24px 22px">
          <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 14px">
            <span style="font-family: IBM Plex Mono, monospace; font-size: 30px; font-weight: 600; color: #8d9a94; line-height: 1">B</span>
            <span style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #7f8b86">bi-predicted · both ways</span>
          </div>
          <p style="margin: 0 0 12px 0; font-size: 14px; color: #b9c3bf">Predicts from a past <em style="font-style: normal; color: #d6dcd9">and</em> a future frame, averaging the two. Cheapest frame in the stream by a wide margin.</p>
          <p style="margin: 0; font-size: 13px; color: #7f8b86"><strong style="color: #9aa5a1; font-weight: 500">Analogy:</strong> interpolating between two known states. To do it, the encoder must send the future first — which costs latency.</p>
          <div style="margin-top: 16px; padding-top: 14px; border-top: 1px dashed #2a2c2b; font-family: IBM Plex Mono, monospace; font-size: 11.5px; color: #8d9a94">≈ {{ statB }} · {{ shareB }} of all bits</div>
        </div>

      </div>
    </section>

    <section style="padding: 48px 0 0 0">
      <div style="display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #1c2321; padding-bottom: 12px; margin-bottom: 28px">
        <h2 style="font-size: 13px; font-family: IBM Plex Mono, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #62e0a1; margin: 0; font-weight: 500">01 · Encoder controls</h2>
        <span style="font-size: 13px; color: #6b7671">every panel below reacts to these</span>
      </div>

      <div style="display: grid; grid-template-columns: 1.15fr 1fr; gap: 36px; align-items: start">

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 26px 32px">
          <sc-for list="{{ sliders }}" as="sl" hint-placeholder-count="4">
            <div style="opacity: {{ sl.op }}">
              <div style="display: flex; justify-content: space-between; align-items: baseline; font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #7f8b86; margin-bottom: 9px">
                <span style="flex: 1; min-width: 0">{{ sl.label }}</span>
                <span style="flex: 0 0 auto; white-space: nowrap; color: #f0f4f2; font-size: 14px; letter-spacing: 0; text-transform: none">{{ sl.display }}</span>
              </div>
              <div onPointerDown="{{ sl.onDown }}" style="height: 22px; display: flex; align-items: center; cursor: ew-resize; touch-action: none; user-select: none">
                <div data-track="1" style="position: relative; height: 3px; width: 100%; background: #1e2523">
                  <div style="position: absolute; left: 0; top: 0; bottom: 0; background: #62e0a1; width: {{ sl.pct }}"></div>
                  <div style="position: absolute; top: -4px; width: 11px; height: 11px; background: #08090a; border: 2px solid #62e0a1; border-radius: 50%; left: {{ sl.pct }}; transform: translateX(-50%)"></div>
                </div>
              </div>
              <div style="font-family: IBM Plex Mono, monospace; font-size: 10.5px; color: #55605c; margin-top: 7px">{{ sl.hint }}</div>
            </div>
          </sc-for>
        </div>

        <div style="display: flex; flex-direction: column; gap: 18px">
          <div>
            <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #7f8b86; margin-bottom: 9px">rate control</div>
            <div style="display: flex; gap: 8px">
              <sc-for list="{{ rcOptions }}" as="rc" hint-placeholder-count="3">
                <button onClick="{{ rc.onClick }}" style="{{ rc.style }}">{{ rc.label }}</button>
              </sc-for>
            </div>
            <div style="font-family: IBM Plex Mono, monospace; font-size: 10.5px; color: #55605c; margin-top: 7px">{{ rcHint }}</div>
          </div>
          <div style="display: flex; flex-direction: column; gap: 10px; border-top: 1px solid #1c2321; padding-top: 18px">
            <sc-for list="{{ toggles }}" as="tg" hint-placeholder-count="3">
              <button onClick="{{ tg.onClick }}" style="{{ tg.style }}">
                <span style="{{ tg.dotStyle }}"></span>
                <span style="flex: 1; text-align: left">{{ tg.label }}</span>
                <span style="font-size: 10.5px; color: #6b7671; text-align: right; max-width: 210px">{{ tg.note }}</span>
              </button>
            </sc-for>
          </div>
        </div>
      </div>
    </section>

    <section style="padding: 52px 0 0 0">
      <div style="display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #1c2321; padding-bottom: 12px; margin-bottom: 10px; flex-wrap: wrap">
        <h2 style="font-size: 13px; font-family: IBM Plex Mono, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #62e0a1; margin: 0; font-weight: 500">02 · The GOP</h2>
        <span style="font-size: 13px; color: #6b7671; flex: 1">bar height = coded size of that frame · click any frame to inspect it</span>
        <div style="display: flex; gap: 6px">
          <sc-for list="{{ modes }}" as="m" hint-placeholder-count="3">
            <button onClick="{{ m.onClick }}" style="{{ m.style }}">{{ m.label }}</button>
          </sc-for>
        </div>
      </div>

      <div style="font-size: 13.5px; color: #9aa5a1; margin: 16px 0 22px 0; min-height: 44px; max-width: 900px; text-wrap: pretty">{{ modeBlurb }}</div>

      <div style="position: relative; border: 1px solid #1c2321; background: #0b0d0c; padding: 18px 20px 0 20px; overflow-x: auto">
        <div style="position: relative; height: 250px; width: {{ tlWidth }}; min-width: 100%">

          <sc-for list="{{ gopBands }}" as="g" hint-placeholder-count="2">
            <div style="{{ g.style }}">
              <div style="position: absolute; top: 6px; left: 8px; font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.1em; color: #4d5854; white-space: nowrap">{{ g.label }}</div>
            </div>
          </sc-for>

          <svg viewBox="{{ svgBox }}" width="100%" height="250" style="position: absolute; inset: 0; pointer-events: none; overflow: visible">
            <sc-for list="{{ arcs }}" as="a" hint-placeholder-count="6">
              <path d="{{ a.d }}" fill="none" stroke="{{ a.stroke }}" stroke-width="{{ a.w }}" stroke-dasharray="{{ a.dash }}" opacity="{{ a.op }}"></path>
            </sc-for>
          </svg>

          <sc-for list="{{ chips }}" as="f" hint-placeholder-count="25">
            <div onClick="{{ f.onClick }}" style="{{ f.style }}">
              <div style="{{ f.barWrap }}">
                <div style="{{ f.bar }}"></div>
              </div>
              <div style="{{ f.letter }}">{{ f.type }}</div>
              <div style="{{ f.idxStyle }}">{{ f.i }}</div>
              <div style="{{ f.flagStyle }}">{{ f.flag }}</div>
            </div>
          </sc-for>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: #1c2321; border: 1px solid #1c2321; border-top: none">
        <sc-for list="{{ readouts }}" as="r" hint-placeholder-count="4">
          <div style="background: #0b0d0c; padding: 16px 18px">
            <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin-bottom: 8px">{{ r.label }}</div>
            <div style="font-family: IBM Plex Mono, monospace; font-size: 21px; color: {{ r.color }}; line-height: 1.1">{{ r.value }}</div>
            <div style="font-size: 11.5px; color: #55605c; margin-top: 7px; line-height: 1.45">{{ r.note }}</div>
          </div>
        </sc-for>
      </div>
    </section>

    <section style="padding: 52px 0 0 0">
      <div style="display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #1c2321; padding-bottom: 12px; margin-bottom: 26px">
        <h2 style="font-size: 13px; font-family: IBM Plex Mono, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #62e0a1; margin: 0; font-weight: 500">03 · Decode order ≠ display order</h2>
        <span style="font-size: 13px; color: #6b7671; flex: 1">step the decoder and watch the reorder buffer fill</span>
        <button onClick="{{ togglePlay }}" style="{{ playStyle }}">{{ playLabel }}</button>
        <button onClick="{{ stepFwd }}" style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; background: none; border: 1px solid #2a332f; color: #9aa5a1; padding: 7px 13px; cursor: pointer">step ›</button>
        <button onClick="{{ reset }}" style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; background: none; border: 1px solid #2a332f; color: #6b7671; padding: 7px 13px; cursor: pointer">reset</button>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 300px; gap: 28px; align-items: start">
        <div style="border: 1px solid #1c2321; background: #0b0d0c; padding: 20px; overflow-x: auto">
          <div style="display: flex; flex-direction: column; gap: 16px; min-width: 640px">
            <sc-for list="{{ orderRows }}" as="row" hint-placeholder-count="2">
              <div>
                <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin-bottom: 8px; display: flex; gap: 10px">
                  <span>{{ row.label }}</span>
                  <span style="color: #3f4946">{{ row.sub }}</span>
                </div>
                <div style="display: flex; gap: 3px">
                  <sc-for list="{{ row.cells }}" as="c" hint-placeholder-count="25">
                    <div style="{{ c.style }}">{{ c.text }}</div>
                  </sc-for>
                </div>
              </div>
            </sc-for>
            <div style="font-family: IBM Plex Mono, monospace; font-size: 11.5px; color: #6b7671; border-top: 1px solid #1c2321; padding-top: 14px; line-height: 1.7">
              <div>dts {{ dtsLabel }} → pts {{ ptsLabel }} · buffer holds {{ bufNow }} frame(s) now, {{ bufMax }} at peak</div>
              <div style="color: #55605c">{{ latencyNote }}</div>
            </div>
          </div>
        </div>

        <div style="border: 1px solid #1c2321; background: #0b0d0c; padding: 20px">
          <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin-bottom: 14px">why it must be this way</div>
          <p style="margin: 0 0 12px 0; font-size: 13.5px; color: #b9c3bf; text-wrap: pretty">A B frame leans on a frame that comes <em style="font-style: normal; color: #62e0a1">after</em> it on screen. So the encoder ships that future anchor early, and the decoder holds finished pictures back until their turn arrives.</p>
          <p style="margin: 0 0 12px 0; font-size: 13.5px; color: #9aa5a1">That hold is why every container carries two clocks: <span style="font-family: IBM Plex Mono, monospace; color: #d6dcd9">DTS</span> for when to decode, <span style="font-family: IBM Plex Mono, monospace; color: #d6dcd9">PTS</span> for when to show.</p>
          <p style="margin: 0; font-size: 12.5px; color: #6b7671">Set B frames to 0 and the two orders collapse into one — the configuration every low-latency conferencing pipeline ships.</p>
        </div>
      </div>
    </section>

    <section style="padding: 52px 0 0 0">
      <div style="display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #1c2321; padding-bottom: 12px; margin-bottom: 26px">
        <h2 style="font-size: 13px; font-family: IBM Plex Mono, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #62e0a1; margin: 0; font-weight: 500">04 · Inside frame {{ selIdx }}</h2>
        <span style="font-size: 13px; color: #6b7671">motion vectors and residual for the frame you selected above</span>
      </div>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr) 300px; gap: 20px; align-items: start">
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin-bottom: 10px">reference {{ refLabel }}</div>
          <canvas data-canvas="ref" width="512" height="288" style="width: 100%; display: block; border: 1px solid #1c2321; background: #0d100f"></canvas>
        </div>
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #62e0a1; margin-bottom: 10px">frame {{ selIdx }} · {{ selType }} · source</div>
          <canvas data-canvas="cur" width="512" height="288" style="width: 100%; display: block; border: 1px solid #223028; background: #0d100f"></canvas>
        </div>
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin-bottom: 10px">{{ resTitle }}</div>
          <canvas data-canvas="res" width="512" height="288" style="width: 100%; display: block; border: 1px solid #1c2321; background: #0d100f"></canvas>
        </div>
        <div style="border-left: 1px solid #1c2321; padding-left: 20px">
          <div style="font-family: IBM Plex Mono, monospace; font-size: 21px; color: #f0f4f2; line-height: 1.1">{{ selSize }}</div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671; margin: 8px 0 16px 0">coded size</div>
          <p style="margin: 0 0 12px 0; font-size: 13px; color: #b9c3bf; text-wrap: pretty">{{ inspectorText }}</p>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 11.5px; color: #6b7671; line-height: 1.8; border-top: 1px dashed #223028; padding-top: 12px">
            <div>refs: <span style="color: #d6dcd9">{{ selRefs }}</span></div>
            <div>coded intra: <span style="color: #d6dcd9">{{ selIntra }}</span></div>
            <div>referenced by: <span style="color: #d6dcd9">{{ selDeps }}</span></div>
          </div>
        </div>
      </div>
      <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; color: #55605c; margin-top: 14px; display: flex; gap: 22px; flex-wrap: wrap">
        <span><span style="color: #62e0a1">→</span> motion vector per 32px block</span>
        <span><span style="color: #62e0a1">▩</span> bright residual = motion could not explain it, so those coefficients cost bits</span>
        <span>synthetic scene · a real scene cut happens at frame {{ cutFrame }}</span>
      </div>
    </section>

    <section style="padding: 52px 0 0 0">
      <div style="display: flex; align-items: baseline; gap: 16px; border-bottom: 1px solid #1c2321; padding-bottom: 12px; margin-bottom: 26px">
        <h2 style="font-size: 13px; font-family: IBM Plex Mono, monospace; letter-spacing: 0.16em; text-transform: uppercase; color: #62e0a1; margin: 0; font-weight: 500">05 · The same settings, as a pipeline</h2>
        <span style="font-size: 13px; color: #6b7671">these strings track the controls above — copy and run</span>
      </div>

      <div style="display: flex; flex-direction: column; gap: 14px">
        <sc-for list="{{ pipelines }}" as="p" hint-placeholder-count="2">
          <div style="border: 1px solid #1c2321; background: #0b0d0c">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #1c2321">
              <span style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #9aa5a1">{{ p.name }}</span>
              <button onClick="{{ p.onCopy }}" style="{{ p.btnStyle }}">{{ p.btnLabel }}</button>
            </div>
            <pre style="margin: 0; padding: 16px; font-family: IBM Plex Mono, monospace; font-size: 12.5px; line-height: 1.9; color: #cfe8dc; white-space: pre-wrap; word-break: break-word">{{ p.cmd }}</pre>
          </div>
        </sc-for>
      </div>

      <div style="margin-top: 22px; border: 1px solid #1c2321">
        <div style="display: grid; grid-template-columns: 1.1fr 1fr 1fr 1.5fr; gap: 1px; background: #1c2321">
          <div style="background: #10130f; padding: 11px 16px; font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671">concept</div>
          <div style="background: #10130f; padding: 11px 16px; font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671">x264enc</div>
          <div style="background: #10130f; padding: 11px 16px; font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671">nvh264enc</div>
          <div style="background: #10130f; padding: 11px 16px; font-family: IBM Plex Mono, monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: #6b7671">what it costs you</div>
          <sc-for list="{{ propRows }}" as="r" hint-placeholder-count="6">
            <div style="background: #0b0d0c; padding: 13px 16px; font-size: 13px; color: #d6dcd9">{{ r.concept }}</div>
            <div style="background: #0b0d0c; padding: 13px 16px; font-family: IBM Plex Mono, monospace; font-size: 12px; color: #62e0a1">{{ r.sw }}</div>
            <div style="background: #0b0d0c; padding: 13px 16px; font-family: IBM Plex Mono, monospace; font-size: 12px; color: #7fb6d6">{{ r.nv }}</div>
            <div style="background: #0b0d0c; padding: 13px 16px; font-size: 12.5px; color: #8b9691">{{ r.cost }}</div>
          </sc-for>
        </div>
      </div>
    </section>

    <section style="padding: 52px 0 0 0">
      <div style="border-top: 1px solid #1c2321; padding-top: 34px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 34px">
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #62e0a1; margin-bottom: 12px">Take one</div>
          <p style="margin: 0; font-size: 14px; color: #b9c3bf; text-wrap: pretty">GOP size is a trade between bitrate and entry points. Long GOP is efficient and cheap; it also means a player joining mid-stream waits, and a lost frame stays broken until the next I.</p>
        </div>
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #62e0a1; margin-bottom: 12px">Take two</div>
          <p style="margin: 0; font-size: 14px; color: #b9c3bf; text-wrap: pretty">B frames buy you bitrate with latency. Great for VOD and streaming ladders, wrong for conferencing, remote desktop, or anything a human steers in real time.</p>
        </div>
        <div>
          <div style="font-family: IBM Plex Mono, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #62e0a1; margin-bottom: 12px">Take three</div>
          <p style="margin: 0; font-size: 14px; color: #b9c3bf; text-wrap: pretty">Frame type decides what a dropped packet destroys. Lose a B and nobody notices; lose an anchor and everything leaning on it smears until the next refresh.</p>
        </div>
      </div>
    </section>

  </div>
</div>

</x-dc>
<script type="text/x-dc" data-dc-script data-props="{&quot;totalFrames&quot;:{&quot;editor&quot;:&quot;int&quot;,&quot;default&quot;:25,&quot;min&quot;:13,&quot;max&quot;:33,&quot;tsType&quot;:&quot;number&quot;,&quot;section&quot;:&quot;Sequence&quot;},&quot;fps&quot;:{&quot;editor&quot;:&quot;enum&quot;,&quot;default&quot;:&quot;30&quot;,&quot;options&quot;:[&quot;24&quot;,&quot;30&quot;,&quot;60&quot;],&quot;tsType&quot;:&quot;string&quot;,&quot;section&quot;:&quot;Sequence&quot;},&quot;startMode&quot;:{&quot;editor&quot;:&quot;enum&quot;,&quot;default&quot;:&quot;deps&quot;,&quot;options&quot;:[&quot;deps&quot;,&quot;seek&quot;,&quot;drop&quot;],&quot;tsType&quot;:&quot;string&quot;,&quot;section&quot;:&quot;Sequence&quot;}}">
class Component extends DCLogic {
  constructor(p) {
    super(p);
    this.state = {
      gop: 12, bframes: 2, refs: 2, bitrate: 6000,
      closed: true, intraRefresh: false, sceneCut: true,
      rc: 'vbr', mode: this.props.startMode || 'deps',
      step: 0, playing: false, copied: ''
    };
  }

  get total() { return this.props.totalFrames || 25; }
  get fps() { return Number(this.props.fps || 30); }
  get cutFrame() { return Math.min(17, this.total - 4); }

  // ---------- model ----------
  build() {
    const s = this.state, total = this.total, cut = this.cutFrame;
    const frames = [];
    for (let i = 0; i < total; i++) frames.push({ i, type: 'B', refs: [], idr: false, pastOnly: false, cross: false, sceneCut: false, refresh: false, gop: 0 });

    if (s.intraRefresh) {
      frames[0].type = 'I'; frames[0].idr = true;
      for (let i = 1; i < total; i++) {
        const f = frames[i]; f.type = 'P'; f.refresh = true;
        for (let r = 1; r <= s.refs && i - r >= 0; r++) f.refs.push(i - r);
      }
    } else {
      const keys = [];
      let k = 0;
      while (k < total) {
        keys.push(k);
        if (s.sceneCut && cut > k && cut < k + s.gop) k = cut; else k += s.gop;
      }
      for (let gi = 0; gi < keys.length; gi++) {
        const start = keys[gi], end = gi + 1 < keys.length ? keys[gi + 1] : total;
        const fi = frames[start];
        fi.type = 'I'; fi.gop = gi;
        fi.sceneCut = s.sceneCut && start === cut && start !== 0;
        fi.idr = s.closed || start === 0 || fi.sceneCut;
        const anchors = [start];
        for (let a = start + 1 + s.bframes; a < end; a += 1 + s.bframes) anchors.push(a);
        for (let ai = 1; ai < anchors.length; ai++) {
          const f = frames[anchors[ai]];
          f.type = 'P'; f.gop = gi;
          f.refs = anchors.slice(Math.max(0, ai - s.refs), ai);
        }
        for (let ai = 0; ai < anchors.length; ai++) {
          const prev = anchors[ai];
          const last = ai + 1 >= anchors.length;
          const stop = last ? end : anchors[ai + 1];
          const fwd = last ? (end < total ? end : null) : anchors[ai + 1];
          const crossing = last && end < total;
          for (let b = prev + 1; b < stop; b++) {
            const f = frames[b];
            f.type = 'B'; f.gop = gi;
            if (fwd !== null && (!crossing || !s.closed)) { f.refs = [prev, fwd]; f.cross = crossing; }
            else { f.refs = [prev]; f.pastOnly = true; }
          }
        }
      }
    }

    // sizes
    const W = s.rc === 'cbr' ? { I: 3.0, P: 1.3, B: 0.72 } : { I: 6.5, P: 1.5, B: 0.55 };
    let sum = 0;
    frames.forEach(f => {
      let w = W[f.type];
      if (f.sceneCut) w *= 1.35;
      if (f.type === 'P' && f.i === cut && !s.sceneCut && !s.intraRefresh) w *= 5.2;  // unflagged scene change
      if (f.refresh) w *= 1 + 0.9 / Math.max(2, s.gop);
      const n = Math.sin(f.i * 12.9898) * 0.5 + 0.5;
      if (s.rc === 'cqp') w *= 0.78 + 0.44 * n;
      else if (s.rc === 'vbr') w *= 0.9 + 0.2 * n;
      f.w = w; sum += w;
    });
    if (s.rc === 'cqp') {
      frames.forEach(f => { f.size = f.w * 2700; });
    } else {
      const avg = (s.bitrate * 1000 / 8) / this.fps;
      const scale = total * avg / sum;
      frames.forEach(f => { f.size = f.w * scale; });
    }
    const bytes = frames.reduce((a, f) => a + f.size, 0);
    const actualKbps = bytes * 8 * this.fps / total / 1000;

    // decode order
    const anchorsAll = frames.filter(f => f.type !== 'B').map(f => f.i);
    const dec = [];
    let prevA = -1;
    anchorsAll.forEach(a => {
      for (let b = prevA + 1; b < a; b++) if (frames[b].type === 'B' && frames[b].pastOnly) dec.push(b);
      dec.push(a);
      for (let b = prevA + 1; b < a; b++) if (frames[b].type === 'B' && !frames[b].pastOnly) dec.push(b);
      prevA = a;
    });
    for (let b = prevA + 1; b < total; b++) if (frames[b].type === 'B') dec.push(b);
    dec.forEach((i, d) => { frames[i].dec = d; });

    // reorder buffer simulation
    let nextOut = 0, held = [], maxBuf = 0;
    const stepStates = [];
    dec.forEach(i => {
      held.push(i);
      const out = [];
      let moved = true;
      while (moved) {
        moved = false;
        const p = held.indexOf(nextOut);
        if (p >= 0) { held.splice(p, 1); out.push(nextOut); nextOut++; moved = true; }
      }
      maxBuf = Math.max(maxBuf, held.length);
      stepStates.push({ decoded: i, out: out.slice(), held: held.slice() });
    });

    // dependents
    frames.forEach(f => { f.usedBy = []; });
    frames.forEach(f => f.refs.forEach(r => frames[r].usedBy.push(f.i)));

    return { frames, dec, stepStates, maxBuf, actualKbps, bytes };
  }

  closure(frames, i, dir) {
    const seen = new Set(), stack = [i];
    while (stack.length) {
      const n = stack.pop();
      const nx = dir === 'up' ? frames[n].refs : frames[n].usedBy;
      nx.forEach(m => { if (!seen.has(m)) { seen.add(m); stack.push(m); } });
    }
    return seen;
  }

  // ---------- interaction ----------
  set(k, v) { this.setState({ [k]: v }); }

  drag(key, min, max, intg) {
    return e => {
      const track = e.currentTarget.querySelector('[data-track]') || e.currentTarget;
      const move = ev => {
        const r = track.getBoundingClientRect();
        let t = (ev.clientX - r.left) / r.width;
        t = Math.max(0, Math.min(1, t));
        let v = min + t * (max - min);
        v = intg ? Math.round(v) : Math.round(v / 100) * 100;
        this.setState({ [key]: v });
      };
      move(e);
      e.preventDefault();
      const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    };
  }

  componentDidMount() {
    this.off = document.createElement('canvas'); this.off.width = 512; this.off.height = 288;
    this.off2 = document.createElement('canvas'); this.off2.width = 512; this.off2.height = 288;
    this.paint();
    requestAnimationFrame(() => this.paint());
    setTimeout(() => this.paint(), 300);
  }
  componentDidUpdate() { this.paint(); }
  componentWillUnmount() { clearInterval(this.timer); }

  togglePlay = () => {
    if (this.state.playing) { clearInterval(this.timer); this.setState({ playing: false }); return; }
    this.timer = setInterval(() => {
      this.setState(st => ({ step: (st.step + 1) % this.total }));
    }, 620);
    this.setState({ playing: true });
  };
  stepFwd = () => this.setState(st => ({ step: (st.step + 1) % this.total }));
  reset = () => { clearInterval(this.timer); this.setState({ step: 0, playing: false }); };

  // ---------- synthetic scene ----------
  scene(ctx, t) {
    const W = 512, H = 288;
    ctx.fillStyle = t >= this.cutFrame ? '#101314' : '#0e120f';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = 'rgba(255,255,255,0.035)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y <= H; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    if (t < this.cutFrame) {
      ctx.fillStyle = 'rgba(98,224,161,0.14)';
      ctx.fillRect(0, 214, W, 74);
      const x = 24 + t * 17, y = 96 + Math.sin(t * 0.42) * 26;
      ctx.fillStyle = '#62e0a1';
      ctx.fillRect(x, y, 62, 62);
      ctx.fillStyle = 'rgba(214,220,217,0.55)';
      ctx.beginPath(); ctx.arc(430 - t * 4, 60, 15, 0, 7); ctx.fill();
    } else {
      const u = t - this.cutFrame;
      ctx.fillStyle = 'rgba(217,166,86,0.10)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = '#d9a656';
      ctx.fillRect(0, 40 + u * 19, W, 34);
      ctx.fillStyle = 'rgba(214,220,217,0.35)';
      ctx.beginPath(); ctx.arc(150 + u * 9, 210, 26, 0, 7); ctx.fill();
    }
  }

  paint() {
    const m = this.model;
    if (!m) return;
    const sel = this.selIdx;
    const f = m.frames[sel];
    const q = k => document.querySelector('[data-canvas="' + k + '"]');
    const cc = q('cur'), rc = q('ref'), sc = q('res');
    if (!cc || !rc || !sc) return;
    const cx = cc.getContext('2d');
    this.scene(cx, sel);

    const refIdx = f.refs.length ? f.refs[0] : null;
    const rx = rc.getContext('2d');
    rx.clearRect(0, 0, 512, 288);
    if (refIdx === null) {
      rx.fillStyle = '#0d100f'; rx.fillRect(0, 0, 512, 288);
      rx.fillStyle = '#7f8b86'; rx.font = '500 26px IBM Plex Mono, monospace';
      rx.fillText('no reference', 22, 140);
      rx.fillStyle = '#55605c';
      rx.fillText('DPB flushed', 22, 176);
    } else {
      this.scene(rx, refIdx);
    }

    // residual + motion vectors
    const sx = sc.getContext('2d');
    sx.fillStyle = '#0a0c0b'; sx.fillRect(0, 0, 512, 288);
    if (refIdx === null) {
      const o = this.off.getContext('2d');
      this.scene(o, sel);
      sx.globalAlpha = 0.5; sx.drawImage(this.off, 0, 0); sx.globalAlpha = 1;
      sx.strokeStyle = 'rgba(98,224,161,0.45)';
      for (let x = 0; x <= 512; x += 32) { sx.beginPath(); sx.moveTo(x, 0); sx.lineTo(x, 288); sx.stroke(); }
      for (let y = 0; y <= 288; y += 32) { sx.beginPath(); sx.moveTo(0, y); sx.lineTo(512, y); sx.stroke(); }
      sx.fillStyle = '#62e0a1'; sx.font = '500 13px IBM Plex Mono, monospace';
      sx.fillText('every block coded intra', 14, 274);
      return;
    }
    const a = this.off.getContext('2d'), b = this.off2.getContext('2d');
    this.scene(a, sel); this.scene(b, refIdx);
    const A = a.getImageData(0, 0, 512, 288), B = b.getImageData(0, 0, 512, 288);
    const out = sx.createImageData(512, 288);
    for (let i = 0; i < A.data.length; i += 4) {
      const d = (Math.abs(A.data[i] - B.data[i]) + Math.abs(A.data[i + 1] - B.data[i + 1]) + Math.abs(A.data[i + 2] - B.data[i + 2])) / 3;
      const v = Math.min(255, d * 3.2);
      out.data[i] = v * 0.45; out.data[i + 1] = v; out.data[i + 2] = v * 0.7; out.data[i + 3] = 255;
    }
    sx.putImageData(out, 0, 0);

    const cutBetween = (sel >= this.cutFrame) !== (refIdx >= this.cutFrame);
    const bs = 32;
    for (let by = 0; by < 288; by += bs) {
      for (let bx = 0; bx < 512; bx += bs) {
        let sad = 0;
        for (let y = by; y < by + bs; y += 4) {
          for (let x = bx; x < bx + bs; x += 4) {
            const i = (y * 512 + x) * 4;
            sad += Math.abs(A.data[i] - B.data[i]) + Math.abs(A.data[i + 1] - B.data[i + 1]);
          }
        }
        if (sad / 128 < 6) continue;
        const cx0 = bx + bs / 2, cy0 = by + bs / 2;
        if (cutBetween) {
          sx.strokeStyle = 'rgba(217,166,86,0.9)'; sx.lineWidth = 1.4;
          sx.beginPath(); sx.moveTo(cx0 - 6, cy0 - 6); sx.lineTo(cx0 + 6, cy0 + 6);
          sx.moveTo(cx0 + 6, cy0 - 6); sx.lineTo(cx0 - 6, cy0 + 6); sx.stroke();
          continue;
        }
        const dt = sel - refIdx;
        const dx = -17 * dt, dy = -(Math.sin(sel * 0.42) - Math.sin(refIdx * 0.42)) * 26;
        const k = Math.min(1, 26 / Math.max(4, Math.hypot(dx, dy)));
        const ex = cx0 + dx * k, ey = cy0 + dy * k;
        sx.strokeStyle = 'rgba(98,224,161,0.95)'; sx.lineWidth = 1.4;
        sx.beginPath(); sx.moveTo(cx0, cy0); sx.lineTo(ex, ey); sx.stroke();
        const ang = Math.atan2(ey - cy0, ex - cx0);
        sx.beginPath(); sx.moveTo(ex, ey);
        sx.lineTo(ex - 5 * Math.cos(ang - 0.5), ey - 5 * Math.sin(ang - 0.5));
        sx.lineTo(ex - 5 * Math.cos(ang + 0.5), ey - 5 * Math.sin(ang + 0.5));
        sx.closePath(); sx.fillStyle = 'rgba(98,224,161,0.95)'; sx.fill();
      }
    }
  }

  copy(text, key) {
    return () => {
      try { navigator.clipboard.writeText(text); } catch (e) {}
      this.setState({ copied: key });
      setTimeout(() => this.setState(st => st.copied === key ? { copied: '' } : {}), 1600);
    };
  }

  kb(bytes) { return bytes > 900 ? (bytes / 1024).toFixed(1) + ' KB' : Math.round(bytes) + ' B'; }

  // ---------- render values ----------
  renderVals() {
    const s = this.state, total = this.total;
    const m = this.build();
    this.model = m;
    const frames = m.frames;
    const step = Math.min(s.step, m.dec.length - 1);
    const sel = m.dec[step];
    this.selIdx = sel;
    const f = frames[sel];

    const CI = '#62e0a1', CP = '#7fb6d6', CB = '#8d9a94', AMB = '#d9a656', RED = '#e0685f';
    const col = t => t === 'I' ? CI : t === 'P' ? CP : CB;

    // stats
    const byType = { I: [], P: [], B: [] };
    frames.forEach(x => byType[x.type].push(x.size));
    const tot = m.bytes || 1;
    const avg = a => a.length ? this.kb(a.reduce((p, c) => p + c, 0) / a.length) : '—';
    const share = a => Math.round(100 * a.reduce((p, c) => p + c, 0) / tot) + '%';

    // sliders
    const sliders = [
      { key: 'gop', label: 'GOP size / key-int-max', display: s.intraRefresh ? '—' : s.gop + ' frames', hint: s.intraRefresh ? 'ignored while intra-refresh is on' : (s.gop / this.fps).toFixed(2) + 's between entry points', min: 1, max: 30, intg: true, op: s.intraRefresh ? 0.4 : 1 },
      { key: 'bframes', label: 'B frames between anchors', display: String(s.bframes), hint: s.bframes === 0 ? 'decode order = display order' : 'adds ' + (s.bframes / this.fps * 1000).toFixed(0) + 'ms of encoder delay', min: 0, max: 4, intg: true, op: s.intraRefresh ? 0.4 : 1 },
      { key: 'refs', label: 'reference frames', display: String(s.refs), hint: 'DPB depth · more refs = better prediction, more memory', min: 1, max: 5, intg: true, op: 1 },
      { key: 'bitrate', label: 'target bitrate', display: s.rc === 'cqp' ? '~' + Math.round(m.actualKbps) + ' kbps' : s.bitrate + ' kbps', hint: s.rc === 'cqp' ? 'CQP does not target a bitrate — this is the result' : 'budget of ' + this.kb(s.bitrate * 1000 / 8 / this.fps) + ' per frame on average', min: 1000, max: 20000, intg: false, op: s.rc === 'cqp' ? 0.4 : 1 }
    ].map(sl => ({
      ...sl,
      pct: Math.round(100 * (s[sl.key] - sl.min) / (sl.max - sl.min)) + '%',
      onDown: this.drag(sl.key, sl.min, sl.max, sl.intg)
    }));

    const btn = (on, c) => ({
      fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px', letterSpacing: '0.1em', textTransform: 'uppercase',
      padding: '8px 14px', cursor: 'pointer', background: on ? 'rgba(98,224,161,0.12)' : 'none',
      border: '1px solid ' + (on ? (c || CI) : '#2a332f'), color: on ? (c || CI) : '#8b9691'
    });

    const rcOptions = [['cbr', 'CBR'], ['vbr', 'VBR'], ['cqp', 'CQP']].map(([k, l]) => ({
      label: l, onClick: () => this.set('rc', k), style: btn(s.rc === k)
    }));
    const rcHint = s.rc === 'cbr'
      ? 'flat pipe: I frames get squeezed toward the average, so quality dips right after a keyframe'
      : s.rc === 'vbr' ? 'lets I frames spend big and coast on cheap B frames — the usual VOD choice'
      : 'constant quality: size follows the content, and the scene cut spikes the bitrate';

    const toggles = [
      { label: 'closed GOP', on: s.closed, note: s.closed ? 'no frame references across a GOP edge — every I is a clean IDR' : 'open: trailing B frames reach into the next GOP, so that I is not a valid entry point', onClick: () => this.set('closed', !s.closed) },
      { label: 'scene-cut detection', on: s.sceneCut, note: s.sceneCut ? 'inserts an IDR at frame ' + this.cutFrame + ' where the content actually changes' : 'off: frame ' + this.cutFrame + ' stays a P frame and pays a huge residual', onClick: () => this.set('sceneCut', !s.sceneCut) },
      { label: 'intra refresh', on: s.intraRefresh, note: s.intraRefresh ? 'one I-slice column per frame instead of whole keyframes — flat bitrate, slow recovery' : 'periodic keyframes: bitrate spikes, instant recovery', onClick: () => this.set('intraRefresh', !s.intraRefresh) }
    ].map(t => ({
      label: t.label, note: t.note, onClick: t.onClick,
      style: {
        display: 'flex', alignItems: 'center', gap: '12px', width: '100%', textAlign: 'left', cursor: 'pointer',
        background: t.on ? 'rgba(98,224,161,0.06)' : 'none', border: '1px solid ' + (t.on ? '#2c4438' : '#202724'),
        padding: '11px 13px', fontFamily: 'IBM Plex Mono, monospace', fontSize: '12px',
        color: t.on ? '#e6efea' : '#8b9691', lineHeight: 1.45
      },
      dotStyle: { width: '8px', height: '8px', flex: '0 0 8px', borderRadius: '50%', background: t.on ? CI : 'transparent', border: '1px solid ' + (t.on ? CI : '#3a433f') }
    }));

    const modes = [['deps', 'dependencies'], ['seek', 'seek'], ['drop', 'packet loss']].map(([k, l]) => ({
      label: l, onClick: () => this.set('mode', k), style: btn(s.mode === k)
    }));

    // highlight sets
    const up = this.closure(frames, sel, 'up');
    const down = this.closure(frames, sel, 'down');
    let needSet = new Set();
    if (s.mode === 'seek') {
      needSet = new Set(up); needSet.add(sel);
    }

    const modeBlurb = s.mode === 'deps'
      ? 'Frame ' + sel + ' is a ' + f.type + ' frame. Bright arcs are the frames it reads; ' + (f.refs.length ? 'lose any of them and this frame cannot be reconstructed.' : 'it reads nothing, which is exactly what makes it an entry point.')
      : s.mode === 'seek'
        ? 'You asked to display frame ' + sel + '. The decoder cannot start there — it rewinds to the last usable IDR and decodes ' + needSet.size + ' frame(s), throwing ' + Math.max(0, needSet.size - 1) + ' of them away just to build the picture you want. Shorter GOP = faster seek, more bits.'
        : 'Frame ' + sel + ' arrives corrupted or never arrives. ' + (down.size === 0 ? 'Nothing references it, so the damage is one frame and it is gone next tick — this is why B frames are the safe thing to lose.' : down.size + ' later frame(s) predicted from it, so the error propagates until the next IDR resets the reference chain.');

    const CHIPW = 40, GAPX = 6, PITCH = CHIPW + GAPX, TOPARC = 62, BARH = 118;
    const xOf = i => 8 + i * PITCH;

    const chips = frames.map(x => {
      const c = col(x.type);
      let state = 'idle';
      if (x.i === sel) state = 'sel';
      else if (s.mode === 'deps' && f.refs.indexOf(x.i) >= 0) state = 'ref';
      else if (s.mode === 'deps' && up.has(x.i)) state = 'chain';
      else if (s.mode === 'seek' && needSet.has(x.i)) state = 'need';
      else if (s.mode === 'drop' && down.has(x.i)) state = 'bad';
      const dim = (s.mode === 'seek' && !needSet.has(x.i)) || (s.mode === 'deps' && !up.has(x.i) && x.i !== sel && f.refs.indexOf(x.i) < 0);
      const maxSize = Math.max.apply(null, frames.map(z => z.size));
      const h = Math.max(3, Math.round(BARH * x.size / maxSize));
      const dropped = s.mode === 'drop' && x.i === sel;
      return {
        i: x.i, type: x.type,
        flag: x.sceneCut ? 'CUT' : x.idr && x.type === 'I' ? 'IDR' : x.type === 'I' ? 'I·open' : x.pastOnly ? 'past' : x.cross ? 'x-gop' : x.refresh ? '▍' : '',
        onClick: () => this.setState({ step: x.dec }),
        style: {
          position: 'absolute', left: xOf(x.i) + 'px', top: TOPARC + 'px', width: CHIPW + 'px',
          cursor: 'pointer', opacity: dim ? 0.28 : 1, transition: 'opacity .18s'
        },
        barWrap: { height: BARH + 'px', display: 'flex', alignItems: 'flex-end' },
        bar: {
          width: '100%', height: h + 'px',
          background: dropped ? 'rgba(224,104,95,0.25)' : state === 'bad' ? 'rgba(217,166,86,0.28)' : c,
          opacity: state === 'idle' ? 0.55 : 1,
          border: dropped ? '1px solid ' + RED : state === 'bad' ? '1px solid ' + AMB : 'none',
          outline: state === 'sel' ? '1px solid #f0f4f2' : 'none', outlineOffset: '2px'
        },
        letter: {
          fontFamily: 'IBM Plex Mono, monospace', fontSize: '15px', fontWeight: 600, textAlign: 'center',
          marginTop: '9px', color: dropped ? RED : state === 'bad' ? AMB : c,
          textDecoration: dropped ? 'line-through' : 'none'
        },
        idxStyle: { fontFamily: 'IBM Plex Mono, monospace', fontSize: '10px', textAlign: 'center', color: x.i === sel ? '#f0f4f2' : '#4d5854', marginTop: '2px' },
        flagStyle: { fontFamily: 'IBM Plex Mono, monospace', fontSize: '8.5px', letterSpacing: '0.06em', textAlign: 'center', color: x.sceneCut ? AMB : '#3f4946', marginTop: '3px', height: '11px' }
      };
    });

    // gop bands
    const bands = [];
    let cur = -1;
    frames.forEach(x => {
      if (x.gop !== cur) { cur = x.gop; bands.push({ gop: cur, from: x.i, to: x.i }); }
      else bands[bands.length - 1].to = x.i;
    });
    const gopBands = bands.map((b, k) => ({
      label: s.intraRefresh ? 'intra-refresh · no GOP boundaries' : 'GOP ' + b.gop + ' · ' + (b.to - b.from + 1) + 'f',
      style: {
        position: 'absolute', left: (xOf(b.from) - 3) + 'px', width: ((b.to - b.from + 1) * PITCH) + 'px',
        top: '0px', height: (TOPARC + BARH + 56) + 'px',
        background: k % 2 ? 'rgba(255,255,255,0.022)' : 'transparent',
        borderLeft: '1px solid ' + (b.from === 0 ? 'transparent' : '#232b28')
      }
    }));

    const arcs = [];
    frames.forEach(x => {
      x.refs.forEach(r => {
        const hot = s.mode === 'deps' && (x.i === sel || (up.has(x.i) && up.has(r)));
        const x1 = xOf(r) + CHIPW / 2, x2 = xOf(x.i) + CHIPW / 2;
        const span = Math.abs(x2 - x1);
        const y = TOPARC - 4, lift = Math.min(52, 14 + span * 0.32);
        arcs.push({
          d: 'M ' + x1 + ' ' + y + ' Q ' + ((x1 + x2) / 2) + ' ' + (y - lift) + ' ' + x2 + ' ' + y,
          stroke: x.type === 'B' ? (x.i < r ? CI : CB) : CP,
          w: hot ? 1.6 : 1, dash: x.i < r ? '3 3' : '0',
          op: hot ? 0.95 : (s.mode === 'deps' ? 0.13 : 0.2)
        });
      });
    });

    const readouts = [
      { label: 'stream bitrate', value: Math.round(m.actualKbps) + ' kbps', color: '#f0f4f2', note: this.kb(m.bytes) + ' for ' + (total / this.fps).toFixed(2) + 's at ' + this.fps + 'fps' },
      { label: 'random access points', value: String(frames.filter(x => x.idr).length), color: CI, note: s.intraRefresh ? 'only the first frame is a true IDR; recovery comes from refresh columns' : 'a player can join or seek cleanly only here' },
      { label: 'worst-case seek cost', value: (s.intraRefresh ? total : Math.max.apply(null, frames.map(x => this.closure(frames, x.i, 'up').size + 1))) + ' frames', color: AMB, note: 'frames decoded but discarded before the target appears' },
      { label: 'reorder buffer', value: m.maxBuf + ' frames', color: m.maxBuf ? CP : '#f0f4f2', note: m.maxBuf ? '+' + (m.maxBuf / this.fps * 1000).toFixed(0) + 'ms of unavoidable display latency' : 'zero added latency — no B frames in the stream' }
    ];

    // order rows
    const st = m.stepStates[step] || { decoded: 0, out: [], held: [] };
    const cell = (txt, kind) => ({
      text: txt,
      style: {
        width: '30px', height: '30px', flex: '0 0 30px', display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontFamily: 'IBM Plex Mono, monospace', fontSize: '11px',
        background: kind === 'now' ? CI : kind === 'held' ? 'rgba(127,182,214,0.18)' : kind === 'done' ? 'rgba(255,255,255,0.05)' : 'transparent',
        color: kind === 'now' ? '#08090a' : kind === 'held' ? CP : kind === 'done' ? '#8b9691' : '#333b38',
        border: '1px solid ' + (kind === 'now' ? CI : kind === 'held' ? 'rgba(127,182,214,0.4)' : '#1c2321'),
        fontWeight: kind === 'now' ? 600 : 400
      }
    });
    const doneDisp = st.out.length ? st.out[st.out.length - 1] : -1;
    const orderRows = [
      {
        label: 'decode order (DTS)', sub: 'what the bitstream actually delivers',
        cells: m.dec.map((i, d) => cell(frames[i].type + i, d === step ? 'now' : d < step ? 'done' : 'wait'))
      },
      {
        label: 'display order (PTS)', sub: 'what the screen shows',
        cells: frames.map(x => cell(x.type + x.i, x.i === doneDisp ? 'now' : st.held.indexOf(x.i) >= 0 ? 'held' : x.i < st.out.concat([-1]).reduce((a, b) => Math.max(a, b), -1) + 1 ? 'done' : 'wait'))
      }
    ];

    const inspectorText = f.type === 'I'
      ? (f.sceneCut ? 'The content genuinely changed here, so prediction was worthless and the encoder spent a whole keyframe. That spike is the right call — mispredicting a cut costs more.' : 'No reference at all: every block is coded from its neighbours inside this frame. Biggest frame in the GOP, and the only place a decoder can start.')
      : f.type === 'P'
        ? (f.i === this.cutFrame && !s.sceneCut ? 'Detection is off, so this P frame is trying to predict a completely different scene. Almost every block falls back to intra and the frame balloons — a P frame doing an I frame\u2019s job, badly.' : 'Motion vectors point back at frame ' + f.refs.join(', ') + '. Only the residual — the part motion could not explain — costs bits.')
        : (f.pastOnly ? 'Closed GOP: this B frame is not allowed to reach past the upcoming I frame, so it predicts from the past only and compresses a little worse.' : 'Predicted from both sides and averaged, so the residual is tiny. Nothing references it, which makes it both the cheapest frame and the safest one to drop.');

    const openTail = frames.some(x => x.cross);
    const latencyNote = m.maxBuf === 0
      ? 'One in, one out. This is what you ship for conferencing and remote rendering.'
      : 'The decoder is holding ' + m.maxBuf + ' finished picture(s) at peak. Containers carry DTS and PTS precisely to describe this gap.';

    const sw = ['gst-launch-1.0 videotestsrc pattern=ball !', 'video/x-raw,width=1920,height=1080,framerate=' + this.fps + '/1 !',
      'x264enc' + (s.intraRefresh ? ' intra-refresh=true' : ' key-int-max=' + s.gop) + ' bframes=' + (s.intraRefresh ? 0 : s.bframes) + ' ref=' + s.refs +
      (s.rc === 'cqp' ? ' pass=qual quantizer=23' : ' pass=' + (s.rc === 'cbr' ? 'cbr' : 'pass1') + ' bitrate=' + s.bitrate) +
      ' option-string="scenecut=' + (s.sceneCut ? 40 : 0) + ':open-gop=' + (s.closed ? 0 : 1) + '" !',
      'h264parse ! mp4mux ! filesink location=out.mp4'].join(' \\\n  ');
    const nv = ['gst-launch-1.0 videotestsrc pattern=ball !', 'video/x-raw,width=1920,height=1080,framerate=' + this.fps + '/1 !', 'nvh264enc' +
      (s.intraRefresh ? ' gop-size=-1' : ' gop-size=' + s.gop) + ' bframes=' + (s.intraRefresh ? 0 : s.bframes) +
      ' rc-mode=' + (s.rc === 'cqp' ? 'constqp qp-const-i=23' : s.rc) + (s.rc === 'cqp' ? '' : ' bitrate=' + s.bitrate) +
      ' preset=hq zerolatency=' + (s.bframes === 0 ? 'true' : 'false') + ' !', 'h264parse ! mp4mux ! filesink location=out.mp4'].join(' \\\n  ');

    const pipelines = [
      { name: 'software · x264enc', cmd: sw, key: 'sw' },
      { name: 'nvidia · nvh264enc', cmd: nv, key: 'nv' }
    ].map(p => ({
      name: p.name, cmd: p.cmd, onCopy: this.copy(p.cmd, p.key),
      btnLabel: s.copied === p.key ? 'copied' : 'copy',
      btnStyle: btn(s.copied === p.key)
    }));

    const propRows = [
      { concept: 'GOP length', sw: 'key-int-max=' + s.gop, nv: 'gop-size=' + s.gop, cost: 'shorter = faster seek and faster loss recovery, more bits' },
      { concept: 'B frames', sw: 'bframes=' + s.bframes, nv: 'bframes=' + s.bframes, cost: 'fewer bits, but ' + s.bframes + ' frame(s) of reorder latency' },
      { concept: 'reference frames', sw: 'ref=' + s.refs, nv: 'via preset / rc-lookahead', cost: 'better prediction, more DPB memory and decode work' },
      { concept: 'closed / open GOP', sw: 'option-string="open-gop=' + (s.closed ? 0 : 1) + '"', nv: 'closed by default', cost: 'open GOP saves bits; only closed GOP gives clean splice points' },
      { concept: 'scene-cut IDR', sw: 'option-string="scenecut=' + (s.sceneCut ? 40 : 0) + '"', nv: 'internal (adaptive I)', cost: 'stops a P frame from paying I-frame prices on a cut' },
      { concept: 'intra refresh', sw: 'intra-refresh=' + (s.intraRefresh ? 'true' : 'false'), nv: 'gop-size=-1 + slice refresh', cost: 'flat bitrate for lossy links; recovery takes a whole cycle' },
      { concept: 'rate control', sw: s.rc === 'cqp' ? 'pass=qual quantizer=23' : 'pass=' + (s.rc === 'cbr' ? 'cbr' : 'pass1'), nv: 'rc-mode=' + (s.rc === 'cqp' ? 'constqp' : s.rc), cost: 'CBR for pipes, VBR for files, CQP when quality is the contract' }
    ];

    return {
      fpsLabel: this.fps + 'fps', totalLabel: total, cutFrame: this.cutFrame,
      statI: avg(byType.I), statP: avg(byType.P), statB: avg(byType.B),
      shareI: share(byType.I), shareP: share(byType.P), shareB: share(byType.B),
      sliders, rcOptions, rcHint, toggles, modes, modeBlurb,
      chips, gopBands, arcs, readouts,
      tlWidth: (total * PITCH + 20) + 'px', svgBox: '0 0 ' + (total * PITCH + 20) + ' 250',
      orderRows, dtsLabel: String(step), ptsLabel: doneDisp >= 0 ? String(doneDisp) : '—',
      bufNow: st.held.length, bufMax: m.maxBuf, latencyNote,
      togglePlay: this.togglePlay, stepFwd: this.stepFwd, reset: this.reset,
      playLabel: s.playing ? '❙❙ pause' : '▶ play decoder', playStyle: btn(s.playing),
      selIdx: sel, selType: f.type + (f.idr ? ' · IDR' : f.pastOnly ? ' · past-only' : ''),
      refLabel: f.refs.length ? String(f.refs[0]) : '—',
      resTitle: f.refs.length ? 'residual + motion vectors' : 'intra coding',
      selSize: this.kb(f.size), selRefs: f.refs.length ? f.refs.join(', ') : 'none',
      selIntra: f.type === 'I' ? '100%' : f.i === this.cutFrame && !s.sceneCut ? '~90%' : f.type === 'P' ? '~8%' : '~2%',
      selDeps: f.usedBy.length ? f.usedBy.join(', ') : 'nothing',
      inspectorText, pipelines, propRows, openTail
    };
  }
}
</script>
