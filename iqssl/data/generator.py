"""The generative distribution: bits in, labelled IQ buffers out.

::

    bits -> symbol mapping -> pulse shaping ----+
                                                +-> emitter -> channel -> AWGN -> crop -> norm
            CPM phase accumulation -------------+   (fixed      (redrawn
                                                     per device) per buffer)

Reproducibility is the property everything else rests on, and it is stronger
than "seeded". Every draw for sample ``i`` comes from ``sample_stream(seed, i)``
and nothing else, so sample 12345 is the same signal whether the dataset holds a
thousand buffers or ten million, generated in one process or thirty-two. A
dataset that quietly changes between runs does not crash -- it produces a results
table that stops meaning what it says.

Two consequences shape the code below, and both look like over-engineering until
you have been bitten:

**Randomness is drawn, then injected.** The DSP primitives all accept
externally-drawn noise (``unit_steps``, ``unit_noise``) precisely so this module
can draw from a per-sample stream rather than from a shared generator whose state
depends on how many samples preceded it in the call.

**Generation runs in windows aligned to the global index.** Batched convolutions
choose different blocking for different batch sizes, so the same sample computed
in a batch of 8 and in a batch of 64 differed in the last mantissa bits. Aligning
to fixed windows means a sample is always computed alongside the same neighbours,
which is what makes the byte-level dataset hash meaningful.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from iqssl.data.params import (
    ROLLOFF_CHOICES,
    SAMPLE_RATE_HZ,
    GeneratorConfig,
    draw_emitters,
    n_symbols_for,
)
from iqssl.dsp import channel as ch
from iqssl.dsp import modulate as mod
from iqssl.dsp.convert import complex_to_ri
from iqssl.dsp.impair import apply_emitter_chain
from iqssl.dsp.power import add_awgn, measure_snr_db, normalize, signal_power
from iqssl.utils.seed import sample_stream

GEN_CHUNK = 256
"""Generation window, in samples, aligned to the global sample index.

Requests are served by computing whole windows and slicing, so sample ``i`` is
always convolved alongside exactly the same neighbours no matter what range was
asked for. Larger is faster and wastes more work on small requests; 256 keeps a
single-sample query cheap enough for interactive use.
"""


class SyntheticIQGenerator:
    """Draws labelled IQ buffers from the parametric perturbation model."""

    def __init__(self, cfg: GeneratorConfig) -> None:
        self.cfg = cfg
        self.preset = cfg.preset
        self.emitters = draw_emitters(self.preset.emitter, self.preset.n_emitters, cfg.seed)
        self.store_len = self.preset.store_len
        self.n_symbols = n_symbols_for(self.store_len, cfg.sps, cfg.span)

        # Impulse-response geometry is fixed from the *prior's* worst case, not
        # from any batch's realized delays. If the CIR length depended on which
        # samples shared a window, buffer geometry would shift with the request
        # and a sample would stop being reproducible on its own.
        max_delay = max(self.preset.channel.delay_spread_symbols) * cfg.sps
        self._cir_origin, self._cir_len = ch.cir_geometry(max_delay)

    # -- per-sample parameter draws -------------------------------------------

    def _draw(self, index: int) -> dict:
        """Every random quantity for one sample, from that sample's stream alone.

        Order matters and must never be rearranged: the stream is consumed
        sequentially, so inserting a draw in the middle silently rewrites every
        sample downstream of it.
        """
        rng = sample_stream(self.cfg.seed, index)
        cfg, pre = self.cfg, self.preset
        chan = pre.channel

        modulation = str(rng.choice(np.array(cfg.modulations)))
        emitter = self.emitters[int(rng.integers(0, pre.n_emitters))]
        rolloff = float(rng.choice(np.array(ROLLOFF_CHOICES)))

        n_taps = int(rng.choice(np.array(chan.n_taps_choices)))
        params = {
            "index": index,
            "modulation": modulation,
            "emitter_id": emitter.emitter_id,
            "rolloff": rolloff,
            "n_taps": n_taps,
            "snr_db_nominal": float(rng.uniform(*chan.snr_db)),
            "cfo_norm": float(rng.uniform(*chan.cfo_norm)),
            "sro_ppm": float(rng.uniform(*chan.sro_ppm)),
            "timing_offset_samples": float(rng.uniform(*chan.timing_offset_samples)),
            "phase0_rad": float(rng.uniform(0.0, 2 * math.pi)),
            # Drawn for every sample even when n_taps == 1, so the field has
            # nonzero variance under single-tap presets and the stored
            # standardization does not divide by zero.
            "delay_spread_symbols": float(rng.uniform(*chan.delay_spread_symbols)),
        }

        modulator = mod.get_modulator(modulation)
        params["_symbols"] = rng.integers(0, modulator.order, size=self.n_symbols)

        # Bulk randomness, drawn here rather than inside the ops so that the
        # signal depends only on this stream.
        params["_pn_steps"] = rng.standard_normal(self._signal_bound)
        params["_pn_phase0"] = float(rng.uniform(0.0, 2 * math.pi))
        params["_tap_gains"] = rng.standard_normal((n_taps, 2))
        params["_tap_delays"] = rng.uniform(0.0, 1.0, size=n_taps)
        params["_noise"] = rng.standard_normal((2, self.store_len))
        return params

    @property
    def _signal_bound(self) -> int:
        """An upper bound on the shaped signal length, before the channel.

        A single bound covering every modulation, rather than the exact length
        per family: linear modulations run a ``span*sps + 1`` tap RRC in ``full``
        mode, CPM convolves a pulse of at most three symbols. ``phase_noise``
        slices the steps it is given down to the signal it receives, so drawing
        slightly too many costs nothing -- whereas making the count depend on the
        modulation would mean the *number* of values consumed from the stream
        varied, and every draw after it would shift.
        """
        return self.n_symbols * self.cfg.sps + (self.cfg.span + 4) * self.cfg.sps

    # -- generation ------------------------------------------------------------

    def generate(self, start: int, n: int) -> tuple[np.ndarray, pd.DataFrame]:
        """``n`` samples beginning at global index ``start``.

        Returns ``(iq, meta)`` with ``iq`` of shape ``(n, 2, store_len)`` float32.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        first = (start // GEN_CHUNK) * GEN_CHUNK
        last = ((start + n - 1) // GEN_CHUNK + 1) * GEN_CHUNK

        iq_parts: list[np.ndarray] = []
        meta_parts: list[pd.DataFrame] = []
        for base in range(first, last, GEN_CHUNK):
            a, m = self._generate_window(base)
            iq_parts.append(a)
            meta_parts.append(m)

        iq = np.concatenate(iq_parts)
        meta = pd.concat(meta_parts, ignore_index=True)

        lo = start - first
        meta = meta.iloc[lo : lo + n].reset_index(drop=True)
        return iq[lo : lo + n], meta

    def _generate_window(self, base: int) -> tuple[np.ndarray, pd.DataFrame]:
        """One whole aligned window. The unit of reproducible computation."""
        draws = [self._draw(base + k) for k in range(GEN_CHUNK)]

        out = np.empty((GEN_CHUNK, 2, self.store_len), dtype=np.float32)
        realized = np.empty(GEN_CHUNK, dtype=np.float32)

        # Group by (modulation, rolloff): those two decide the pulse-shaping
        # taps, and a shared kernel is what allows one batched convolution per
        # group instead of one per sample.
        groups: dict[tuple[str, float], list[int]] = {}
        for k, d in enumerate(draws):
            groups.setdefault((d["modulation"], d["rolloff"]), []).append(k)

        for (modulation, rolloff), idx in groups.items():
            buf, snr = self._render_group(modulation, rolloff, [draws[k] for k in idx])
            out[idx] = buf
            realized[idx] = snr

        meta = pd.DataFrame([{k: v for k, v in d.items() if not k.startswith("_")} for d in draws])
        meta["snr_db_realized"] = realized
        for f in ("iq_gain_db", "iq_phase_deg", "dc_i_dbc", "dc_q_dbc", "pa_ibo_db"):
            meta[f"emitter_{f}"] = [getattr(self.emitters[d["emitter_id"]], f) for d in draws]
        return out, meta

    def _render_group(
        self, modulation: str, rolloff: float, draws: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        b = len(draws)
        modulator = mod.get_modulator(modulation)

        symbols = torch.from_numpy(np.stack([d["_symbols"] for d in draws])).long()
        # CPM modulators accept and ignore rolloff/span: pulse-shaping a
        # constant-modulus signal would destroy the property that makes the
        # family behave differently under PA compression.
        x = modulator.modulate(symbols, sps=cfg.sps, rolloff=rolloff, span=cfg.span, mode="full")

        x = self._apply_emitter(x, draws)
        x = self._apply_channel(x, draws)

        # Crop *before* adding noise, and measure the power of exactly the
        # samples being kept. PA compression and the fading realization both
        # move power by several dB, so noise scaled to the pre-channel power
        # would leave the stored SNR label describing a signal that no longer
        # exists.
        clean = self._crop(x)
        unit = torch.from_numpy(np.stack([d["_noise"] for d in draws])).float()
        noisy, _ = add_awgn(
            clean,
            torch.tensor([d["snr_db_nominal"] for d in draws], dtype=torch.float32),
            measured_power=signal_power(clean),
            unit_noise=torch.complex(unit[:, 0], unit[:, 1]),
        )
        realized = measure_snr_db(clean, noisy)

        y = normalize(noisy, cfg.normalize)
        assert y.shape == (b, self.store_len)
        return complex_to_ri(y).numpy().astype(np.float32), realized.numpy()

    def _apply_emitter(self, x: torch.Tensor, draws: list[dict]) -> torch.Tensor:
        em = [self.emitters[d["emitter_id"]] for d in draws]

        def col(name: str) -> torch.Tensor:
            return torch.tensor([getattr(e, name) for e in em], dtype=torch.float32)

        steps = torch.from_numpy(np.stack([d["_pn_steps"][: x.shape[1]] for d in draws])).float()
        return apply_emitter_chain(
            x,
            iq_gain_db=col("iq_gain_db"),
            iq_phase_deg=col("iq_phase_deg"),
            dc_i_dbc=col("dc_i_dbc"),
            dc_q_dbc=col("dc_q_dbc"),
            pa_ibo_db=col("pa_ibo_db"),
            pn_linewidth_hz=col("pn_linewidth_hz"),
            sample_rate_hz=SAMPLE_RATE_HZ,
            pn_unit_steps=steps,
            pn_initial_phase=torch.tensor([d["_pn_phase0"] for d in draws], dtype=torch.float32),
        )

    def _apply_channel(self, x: torch.Tensor, draws: list[dict]) -> torch.Tensor:
        b = len(draws)
        # Samples in a group may differ in tap count, so the tap dimension is
        # padded to the widest and the unused taps stay at zero gain, which
        # contributes nothing to the impulse response.
        n_taps = max(d["n_taps"] for d in draws)

        gains = torch.zeros(b, n_taps, dtype=torch.complex64)
        delays = torch.zeros(b, n_taps, dtype=torch.float32)
        for i, d in enumerate(draws):
            t = d["n_taps"]
            # Exponential power delay profile, normalized so the average channel
            # gain is unity and fading does not covertly shift SNR.
            p = ch.exponential_pdp(t)
            g = torch.from_numpy(d["_tap_gains"]).float()
            cn = torch.complex(g[:, 0], g[:, 1]) / math.sqrt(2.0)
            gains[i, :t] = torch.sqrt(p) * cn
            if t > 1:
                spread = d["delay_spread_symbols"] * self.cfg.sps
                dl = np.sort(d["_tap_delays"]) * spread
                dl[0] = 0.0  # first arrival defines the time reference
                delays[i, :t] = torch.from_numpy(dl).float()

        cir = ch.build_cir(gains, delays, origin=self._cir_origin, out_len=self._cir_len)
        y = ch.filter_per_sample(x, cir)

        y = ch.apply_cfo(
            y,
            torch.tensor([d["cfo_norm"] for d in draws], dtype=torch.float32),
            torch.tensor([d["phase0_rad"] for d in draws], dtype=torch.float32),
        )
        y = ch.apply_fractional_delay(
            y, torch.tensor([d["timing_offset_samples"] for d in draws], dtype=torch.float32)
        )
        return ch.apply_sample_rate_offset(
            y, torch.tensor([d["sro_ppm"] for d in draws], dtype=torch.float32)
        )

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        """Trim filter transients, then take ``store_len`` samples.

        The offset is a fixed distance past the pulse-shaping transient rather
        than the buffer centre: a ramp-up sitting at a constant position in every
        buffer is exactly the kind of shortcut feature a network learns instead
        of the task.
        """
        start = self.cfg.span * self.cfg.sps
        end = start + self.store_len
        if end > x.shape[1]:
            raise RuntimeError(
                f"signal is {x.shape[1]} samples but the crop needs {end}; "
                "n_symbols_for() and _signal_len() disagree"
            )
        return x[:, start:end]
