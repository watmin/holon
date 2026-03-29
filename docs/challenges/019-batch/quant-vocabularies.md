# Batch 019 — Quantitative Vocabulary Specifications

Six statistical/quantitative indicator vocabularies for the thought encoder.
Each is self-contained: atoms, computation, zones, comparison pairs, predicates,
and rationale for why quants use it and retail doesn't.

All computations operate on the 48-candle window already available in `encode_view()`.
All atoms follow existing naming conventions and encoding rules from THOUGHT_VOCAB.md.

---

## 1. Z-Score Normalization

### Why quants use this and retail doesn't

Retail traders see "price is above the 20-SMA" — a binary fact with no magnitude.
Quants see "price is 2.3 sigma above the 20-SMA" — a precise statement about how
extreme the deviation is. Z-score normalizes distance by volatility, making
thresholds regime-independent. A $500 deviation means nothing without knowing if
daily range is $200 or $5000. Z-score tells you whether $500 is noise or a
three-sigma event. This is the single most important normalization in quantitative
finance and the foundation of mean-reversion strategies (Bollinger Bands are just
z-score with a pretty wrapper).

### Atoms

```
;; ─── NEW INDICATORS (3) ─────────────────────────────────────────
zscore-20        ;; (close - sma20) / std20
zscore-50        ;; (close - sma50) / std50
zscore-200       ;; (close - sma200) / std200

;; ─── NEW ZONES (6) ──────────────────────────────────────────────
z-extreme-high   ;; zscore > +2.0   (top ~2.3% of normal dist)
z-high           ;; +1.0 < zscore <= +2.0
z-neutral        ;; -1.0 <= zscore <= +1.0 (68% of observations)
z-low            ;; -2.0 <= zscore < -1.0
z-extreme-low    ;; zscore < -2.0
z-reverting      ;; |zscore| was > 1.5, now < 1.0 (mean reversion in progress)
```

### Computation from 48-candle window

```rust
fn compute_zscore(candles: &[Candle], sma_field: fn(&Candle) -> f64) -> f64 {
    let n = candles.len();
    let now = &candles[n - 1];
    let close = now.close;
    let sma = sma_field(now);
    if sma <= 0.0 { return 0.0; }

    // Standard deviation of close prices over the window
    let mean: f64 = candles.iter().map(|c| c.close).sum::<f64>() / n as f64;
    let variance: f64 = candles.iter()
        .map(|c| (c.close - mean).powi(2))
        .sum::<f64>() / n as f64;
    let std = variance.sqrt();
    if std < 1e-10 { return 0.0; }

    (close - sma) / std
}

// For zscore-20: use last 20 candles, sma = candle.sma20
// For zscore-50: use full 48-candle window, sma = candle.sma50
// For zscore-200: use full window, sma = candle.sma200
//   (note: std200 approximated from 48 candles, not 200 — acceptable
//    because we care about recent volatility regime, not historical)
```

### Zone thresholds

| Zone            | Condition                                        |
|-----------------|--------------------------------------------------|
| z-extreme-high  | zscore > 2.0                                     |
| z-high          | 1.0 < zscore <= 2.0                              |
| z-neutral       | -1.0 <= zscore <= 1.0                            |
| z-low           | -2.0 <= zscore < -1.0                            |
| z-extreme-low   | zscore < -2.0                                    |
| z-reverting     | prev zscore abs > 1.5 AND current zscore abs < 1.0 |

### Comparison pairs

```
("zscore-20", "zscore-50"),      ;; short vs medium term deviation
("zscore-50", "zscore-200"),     ;; medium vs long term deviation
```

### Predicates (use existing)

```
(at zscore-20 z-extreme-high)    ;; price is 2+ sigma above 20-SMA
(at zscore-20 z-extreme-low)     ;; mean-reversion buy zone
(at zscore-20 z-reverting)       ;; mean reversion in progress
(above zscore-20 zscore-50)      ;; short-term more extended than medium
(crosses-below zscore-20 zscore-50) ;; short-term deviation collapsing
```

### Expert profile

Assign to: `"momentum"` (z-score is a momentum/mean-reversion signal).

### Implementation notes

- zscore-20/50/200 are derived indicators (computed from window, not DB columns).
  Add to `candle_field()` match or compute in a dedicated `eval_zscore()` method.
- The z-reverting zone requires prev-candle zscore. Store zscore in a field or
  recompute from candles[0..n-1].

---

## 2. Autocorrelation of Returns

### Why quants use this and retail doesn't

Retail traders look at price direction. Quants look at whether price direction
is *persistent*. Positive autocorrelation means today's return predicts tomorrow's
direction (trending market — momentum strategies work). Negative autocorrelation
means today's return predicts the *opposite* tomorrow (mean-reverting market —
fade strategies work). Near-zero means returns are independent (random walk —
neither strategy has edge). This is the mathematical foundation for deciding
between momentum and mean-reversion strategies. Hedge funds like AQR and
Two Sigma estimate autocorrelation regimes in real-time. Most retail traders
have never heard the word.

### Atoms

```
;; ─── NEW INDICATORS (2) ─────────────────────────────────────────
autocorr-1       ;; lag-1 autocorrelation of log returns (trending vs reverting)
autocorr-5       ;; lag-5 autocorrelation (medium-term persistence)

;; ─── NEW ZONES (5) ──────────────────────────────────────────────
strongly-trending    ;; autocorr-1 > +0.3 (persistent momentum)
weakly-trending      ;; +0.1 < autocorr-1 <= +0.3
random-walk          ;; -0.1 <= autocorr-1 <= +0.1 (no exploitable pattern)
weakly-reverting     ;; -0.3 <= autocorr-1 < -0.1
strongly-reverting   ;; autocorr-1 < -0.3 (strong mean reversion)
```

### Computation from 48-candle window

```rust
fn compute_autocorrelation(candles: &[Candle], lag: usize) -> f64 {
    let n = candles.len();
    if n < lag + 5 { return 0.0; }

    // Log returns
    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let r_n = returns.len();
    if r_n < lag + 2 { return 0.0; }

    let mean: f64 = returns.iter().sum::<f64>() / r_n as f64;

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for i in 0..r_n {
        let dev = returns[i] - mean;
        den += dev * dev;
        if i >= lag {
            num += dev * (returns[i - lag] - mean);
        }
    }
    if den < 1e-20 { return 0.0; }
    num / den
}
```

### Zone thresholds

| Zone               | Condition                    | Meaning                         |
|--------------------|------------------------------|---------------------------------|
| strongly-trending  | autocorr-1 > 0.3            | Momentum strategies profitable  |
| weakly-trending    | 0.1 < autocorr-1 <= 0.3     | Mild momentum edge              |
| random-walk        | -0.1 <= autocorr-1 <= 0.1   | No directional edge             |
| weakly-reverting   | -0.3 <= autocorr-1 < -0.1   | Mild mean-reversion edge        |
| strongly-reverting | autocorr-1 < -0.3           | Fade strategies profitable      |

### Comparison pairs

```
("autocorr-1", "autocorr-5"),    ;; short vs medium persistence
```

### Predicates

```
(at autocorr-1 strongly-trending)     ;; momentum regime detected
(at autocorr-1 strongly-reverting)    ;; mean-reversion regime detected
(at autocorr-1 random-walk)           ;; no edge — reduce position size
(above autocorr-1 autocorr-5)         ;; short-term more persistent than medium
```

### Expert profile

Assign to: `"momentum"` (this IS the regime classifier for momentum vs reversion).

### Implementation notes

- Autocorrelation is computed from the window of returns, not from a single candle.
  Must be computed fresh each call in `eval_autocorrelation()`.
- The lag-1 autocorrelation on 5-minute BTC data will typically be slightly negative
  (bid-ask bounce) or slightly positive (momentum). Values > 0.3 or < -0.3 are
  genuinely informative regime signals.
- Consider also emitting a regime-change fact when autocorr-1 crosses zero
  (switches from trending to reverting or vice versa).

---

## 3. Entropy of Price Changes

### Why quants use this and retail doesn't

Entropy answers the question retail traders never ask: "Is the market even
predictable right now?" High entropy means price changes are uniformly distributed
across bins — the market is maximally random and NO strategy has edge. Low entropy
means price changes cluster in specific bins — there's structure to exploit.
Market makers use entropy to adjust spread widths. Quant funds use it as a
meta-signal to scale position sizes. This is information theory applied to markets,
and it separates "I can't find a pattern" (trader problem) from "there is no
pattern" (market property). Shannon entropy on discretized returns is the
canonical measure.

### Atoms

```
;; ─── NEW INDICATORS (1) ─────────────────────────────────────────
return-entropy   ;; Shannon entropy of discretized returns (0.0 to 1.0, normalized)

;; ─── NEW ZONES (4) ──────────────────────────────────────────────
high-entropy     ;; entropy > 0.85 (near-random, don't trade)
medium-entropy   ;; 0.60 < entropy <= 0.85 (some structure)
low-entropy      ;; 0.35 < entropy <= 0.60 (structured, tradeable)
very-low-entropy ;; entropy <= 0.35 (highly structured — strong patterns)
```

### Computation from 48-candle window

```rust
fn compute_return_entropy(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 10 { return 1.0; } // assume random if insufficient data

    // Log returns
    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    // Discretize into 7 bins based on ATR-normalized magnitude:
    //   [-inf, -1.5], (-1.5, -0.75], (-0.75, -0.25], (-0.25, 0.25],
    //   (0.25, 0.75], (0.75, 1.5], (1.5, +inf]
    // Normalize returns by median absolute return for regime independence.
    let abs_returns: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    let mut sorted_abs = abs_returns.clone();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_abs = sorted_abs[sorted_abs.len() / 2].max(1e-10);

    let mut bins = [0u32; 7];
    for &r in &returns {
        let normalized = r / median_abs;
        let bin = if normalized < -1.5 { 0 }
            else if normalized < -0.75 { 1 }
            else if normalized < -0.25 { 2 }
            else if normalized <  0.25 { 3 }
            else if normalized <  0.75 { 4 }
            else if normalized <  1.50 { 5 }
            else { 6 };
        bins[bin] += 1;
    }

    // Shannon entropy, normalized to [0, 1]
    let total = returns.len() as f64;
    let max_entropy = (7.0_f64).ln(); // ln(num_bins)
    let entropy: f64 = bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / total;
            -p * p.ln()
        })
        .sum();

    entropy / max_entropy // normalized: 0 = one bin dominates, 1 = uniform
}
```

### Zone thresholds

| Zone             | Condition       | Trading implication                         |
|------------------|-----------------|---------------------------------------------|
| high-entropy     | entropy > 0.85  | Market is near-random. Reduce position size. |
| medium-entropy   | 0.60 - 0.85     | Some structure. Normal trading.              |
| low-entropy      | 0.35 - 0.60     | Structured. Increase conviction.             |
| very-low-entropy | entropy <= 0.35  | Strong pattern. Rare — high-conviction signal.|

### Comparison pairs

None — entropy is a scalar regime descriptor, not compared pairwise with
other indicators. It modulates *how much* to trust other signals.

### Predicates

```
(at return-entropy high-entropy)       ;; market is random — don't trade
(at return-entropy very-low-entropy)   ;; strong structure — trade confidently
(at return-entropy low-entropy)        ;; structure present — normal edge
```

### Expert profile

Assign to: `"structure"` (entropy describes market structure/regime).

### Implementation notes

- This is a META-signal. It doesn't say BUY or SELL. It says "your other signals
  are meaningful" or "your other signals are noise." The orchestrator could use
  entropy zone to scale conviction.
- The 7-bin discretization is a design choice. Fewer bins = less sensitivity to
  small variations, more stable. More bins = finer resolution but noisier with
  only 47 returns. Seven bins with 47 data points gives ~6.7 samples/bin on
  average under uniform distribution — statistically adequate.
- Normalize by median absolute return, NOT by ATR. ATR uses high-low range which
  conflates intrabar volatility with close-to-close variation. Median absolute
  return is the self-consistent normalizer.

---

## 4. Donchian Channels

### Why quants use this and retail doesn't

Richard Dennis used Donchian Channels to train the Turtle Traders in the 1980s —
possibly the most successful trading experiment in history. The concept is
brutally simple: buy when price breaks the N-period highest high; sell when it
breaks the N-period lowest low. Retail traders dismiss it as "too simple" and
chase complex indicators. But Donchian channels encode a fact that matters deeply:
where is the current price relative to its FULL range? Not relative to a moving
average (which lags), but relative to the actual extremes. A breakout above the
20-period high is an objective, unambiguous event. There is no parameter to
tune except N. This simplicity is the strength — it's nearly impossible to
overfit. Quant trend-following funds (Man AHL, Winton, etc.) still use channel
breakouts as core signals.

### Atoms

```
;; ─── NEW INDICATORS (5) ─────────────────────────────────────────
donchian-high-20     ;; 20-period highest high
donchian-low-20      ;; 20-period lowest low
donchian-mid-20      ;; (donchian-high-20 + donchian-low-20) / 2
donchian-width-20    ;; (donchian-high-20 - donchian-low-20) / close (normalized)
donchian-pos-20      ;; (close - donchian-low-20) / (donchian-high-20 - donchian-low-20)
                     ;; Position within channel: 0.0 = at low, 1.0 = at high

;; ─── NEW ZONES (6) ──────────────────────────────────────────────
donchian-breakout-high  ;; close >= donchian-high-20 (new 20-period high)
donchian-breakout-low   ;; close <= donchian-low-20 (new 20-period low)
donchian-upper-zone     ;; donchian-pos-20 > 0.80 (near top of range)
donchian-lower-zone     ;; donchian-pos-20 < 0.20 (near bottom of range)
donchian-mid-zone       ;; 0.35 <= donchian-pos-20 <= 0.65 (range-bound middle)
donchian-narrowing      ;; donchian-width-20 contracting (current < prev by > 10%)
```

### Computation from 48-candle window

```rust
fn compute_donchian(candles: &[Candle], period: usize) -> (f64, f64) {
    // Use last `period` candles from the window
    let n = candles.len();
    let start = if n > period { n - period } else { 0 };
    let slice = &candles[start..n];

    let high = slice.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let low = slice.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    (high, low)
}

fn donchian_position(close: f64, high: f64, low: f64) -> f64 {
    let range = high - low;
    if range < 1e-10 { return 0.5; }
    ((close - low) / range).clamp(0.0, 1.0)
}

fn donchian_width_normalized(high: f64, low: f64, close: f64) -> f64 {
    if close < 1e-10 { return 0.0; }
    (high - low) / close
}
```

### Zone thresholds

| Zone                  | Condition                              |
|-----------------------|----------------------------------------|
| donchian-breakout-high| close >= donchian-high-20              |
| donchian-breakout-low | close <= donchian-low-20               |
| donchian-upper-zone   | donchian-pos-20 > 0.80                 |
| donchian-lower-zone   | donchian-pos-20 < 0.20                 |
| donchian-mid-zone     | 0.35 <= donchian-pos-20 <= 0.65        |
| donchian-narrowing    | current width < width[5 bars ago] * 0.9|

### Comparison pairs

```
("close", "donchian-high-20"),   ;; proximity to channel top
("close", "donchian-low-20"),    ;; proximity to channel bottom
("close", "donchian-mid-20"),    ;; above/below channel midpoint
("donchian-width-20", "bb-width"), ;; Donchian vs Bollinger width (volatility agreement)
```

### Predicates

```
(at close donchian-breakout-high)    ;; Turtle buy signal
(at close donchian-breakout-low)     ;; Turtle sell signal
(at donchian-pos-20 donchian-upper-zone)  ;; near top, potential resistance
(at donchian-pos-20 donchian-lower-zone)  ;; near bottom, potential support
(at donchian-width-20 donchian-narrowing) ;; range compression (breakout imminent)
(above close donchian-mid-20)        ;; bullish channel position
(below close donchian-mid-20)        ;; bearish channel position
(crosses-above close donchian-high-20) ;; breakout event
(crosses-below close donchian-low-20)  ;; breakdown event
```

### Expert profile

Assign to: `"structure"` (channel position is structural context).

### Implementation notes

- Donchian channels use high/low (not close), which makes them immune to the
  noise that afflicts close-only indicators. This is a feature.
- The breakout zones (close >= donchian-high-20) will fire on the exact candle
  that sets a new 20-period high. This is a one-candle event. The upper-zone
  (pos > 0.80) provides a softer "approaching breakout" signal.
- donchian-narrowing serves the same conceptual role as Bollinger squeeze but
  is computed differently (range of extremes vs standard deviation). When both
  donchian-narrowing and squeeze fire simultaneously, that's a very strong
  volatility compression signal the journaler can discover.
- Period 20 matches the original Turtle system. Adding period 55 (the Turtle
  exit period) is a natural extension but not included here to keep atom count
  manageable.

---

## 5. Ultimate Oscillator

### Why quants use this and retail doesn't

Single-period oscillators (RSI-14, Stochastic-14) give false signals because
they only see one timeframe. Larry Williams designed the Ultimate Oscillator to
combine three timeframes (7, 14, 28 periods) into one number, weighted to favor
the shortest period. The key insight: when a move is real, all three timeframes
agree. When it's noise, only the short period reacts. This multi-scale agreement
test is exactly what quants want — it's a poor man's wavelet decomposition of
momentum. Retail traders either use RSI (one period) or try to manually reconcile
multiple RSIs on different charts. The Ultimate Oscillator does this algebraically
in a single number. It also uses True Range and Buying Pressure (close - true low),
not just close-to-close changes, so it captures intrabar dynamics that RSI misses.

### Atoms

```
;; ─── NEW INDICATORS (1) ─────────────────────────────────────────
ult-osc          ;; Ultimate Oscillator value (0 to 100)

;; ─── NEW ZONES (5) ──────────────────────────────────────────────
ult-overbought   ;; ult-osc > 70
ult-oversold     ;; ult-osc < 30
ult-bullish      ;; ult-osc > 50 (buying pressure dominant)
ult-bearish      ;; ult-osc <= 50 (selling pressure dominant)
ult-extreme-low  ;; ult-osc < 20 (rare — extreme oversold)
```

### Computation from 48-candle window

```rust
fn compute_ultimate_oscillator(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 29 { return 50.0; } // need 28 periods + 1 for true range

    // Buying Pressure (BP) and True Range (TR) for each candle
    let mut bp = Vec::with_capacity(n);
    let mut tr = Vec::with_capacity(n);

    for i in 1..n {
        let prev_close = candles[i - 1].close;
        let true_low = candles[i].low.min(prev_close);
        let true_high = candles[i].high.max(prev_close);
        bp.push(candles[i].close - true_low);
        tr.push(true_high - true_low);
    }

    // Sum BP and TR over three periods (from the end)
    let len = bp.len();

    let sum_bp = |period: usize| -> f64 {
        bp[len - period..].iter().sum()
    };
    let sum_tr = |period: usize| -> f64 {
        let s: f64 = tr[len - period..].iter().sum();
        s.max(1e-10)
    };

    let avg7  = sum_bp(7)  / sum_tr(7);
    let avg14 = sum_bp(14) / sum_tr(14);
    let avg28 = sum_bp(28) / sum_tr(28);

    // Weighted: 4x short, 2x medium, 1x long (Williams' original weights)
    let uo = 100.0 * (4.0 * avg7 + 2.0 * avg14 + avg28) / 7.0;
    uo.clamp(0.0, 100.0)
}
```

### Zone thresholds

| Zone            | Condition      | Meaning                                   |
|-----------------|----------------|-------------------------------------------|
| ult-overbought  | ult-osc > 70   | Multi-timeframe buying exhaustion         |
| ult-oversold    | ult-osc < 30   | Multi-timeframe selling exhaustion        |
| ult-bullish     | ult-osc > 50   | Net buying pressure across all periods    |
| ult-bearish     | ult-osc <= 50  | Net selling pressure across all periods   |
| ult-extreme-low | ult-osc < 20   | Capitulation-level selling (rare, strong) |

### Comparison pairs

```
("ult-osc", "rsi"),              ;; multi-period vs single-period momentum
```

Note: direct comparison between UO and RSI is meaningful because both are on
0-100 scale. When UO > RSI, multi-timeframe momentum is stronger than single-
timeframe — the move has broad-based support. When RSI > UO, the move is
driven by the short-term only — potentially fading.

### Predicates

```
(at ult-osc ult-overbought)          ;; multi-TF overbought
(at ult-osc ult-oversold)            ;; multi-TF oversold
(at ult-osc ult-extreme-low)         ;; capitulation
(above ult-osc rsi)                  ;; broad momentum > single-period
(below ult-osc rsi)                  ;; narrow momentum only
(crosses-above ult-osc rsi)          ;; momentum broadening
(crosses-below ult-osc rsi)          ;; momentum narrowing

;; Williams' original buy signal (divergence):
;; - ult-osc makes higher low while price makes lower low
;; - ult-osc was below 30 during the divergence
;; This emerges naturally from existing divergence predicates:
(diverging close down ult-osc up)    ;; bullish UO divergence
(diverging close up ult-osc down)    ;; bearish UO divergence
```

### Expert profile

Assign to: `"momentum"` (this is a momentum oscillator).

### Implementation notes

- The Ultimate Oscillator requires prev_close for True Range/True Low, so
  computation starts at candle index 1. With a 48-candle window, we have 47
  data points — more than enough for the 28-period lookback.
- Williams' original trading rules include a divergence requirement for buy
  signals. This doesn't need to be hardcoded — the divergence predicate
  `(diverging close down ult-osc up)` combined with `(at ult-osc ult-oversold)`
  will be discovered by the journaler if it's predictive.
- The weighting (4:2:1) is not arbitrary — it ensures the oscillator is most
  responsive to recent action while still requiring confirmation from longer
  periods. Do not change these weights.

---

## 6. Mass Index

### Why quants use this and retail doesn't

The Mass Index is specifically designed to detect "reversal bulges" — moments
when the range between high and low expands and then contracts, which often
precedes a reversal. It was created by Donald Dorsey for exactly one purpose:
catching reversals before they happen. Retail traders have never heard of it
because it doesn't appear in popular TA books, it has no flashy chart overlay,
and its output doesn't say "buy" or "sell" — it says "something is about to
change." This ambiguity makes it useless for retail's directional bias, but
incredibly valuable as a regime-change detector. Quant funds use similar range
expansion/contraction metrics (often proprietary variants) as early warning
systems. The Mass Index is the published, well-tested version.

### Atoms

```
;; ─── NEW INDICATORS (1) ─────────────────────────────────────────
mass-index       ;; 25-period sum of EMA9(range) / EMA9(EMA9(range))

;; ─── NEW ZONES (3) ──────────────────────────────────────────────
mass-bulge       ;; mass-index crossed above 27.0 then back below 26.5
                 ;; (the "reversal bulge" — Dorsey's original signal)
mass-expanding   ;; mass-index > 27.0 (range expanding, reversal building)
mass-contracting ;; mass-index < 25.0 (range compressed, stable trend)
```

### Computation from 48-candle window

```rust
fn compute_mass_index(candles: &[Candle]) -> (f64, bool) {
    // Returns (mass_index_value, bulge_detected)
    let n = candles.len();
    if n < 35 { return (25.0, false); } // need enough data for double EMA + sum

    // Step 1: High-Low range for each candle
    let ranges: Vec<f64> = candles.iter().map(|c| c.high - c.low).collect();

    // Step 2: EMA-9 of the range (single smoothing)
    let k = 2.0 / 10.0; // EMA period 9: k = 2/(9+1)
    let mut ema1 = Vec::with_capacity(n);
    ema1.push(ranges[0]);
    for i in 1..n {
        ema1.push(ranges[i] * k + ema1[i - 1] * (1.0 - k));
    }

    // Step 3: EMA-9 of EMA-9 (double smoothing)
    let mut ema2 = Vec::with_capacity(n);
    ema2.push(ema1[0]);
    for i in 1..n {
        ema2.push(ema1[i] * k + ema2[i - 1] * (1.0 - k));
    }

    // Step 4: Ratio = single_ema / double_ema, then sum last 25
    let start = if n > 25 { n - 25 } else { 0 };
    let mass_index: f64 = (start..n)
        .map(|i| {
            if ema2[i] < 1e-10 { 1.0 } else { ema1[i] / ema2[i] }
        })
        .sum();

    // Step 5: Detect bulge — did mass_index cross above 27 recently
    // and is now back below 26.5?
    // Approximate: check if current < 26.5 and any of last 10 values > 27.0
    let bulge = if mass_index < 26.5 {
        // Recompute mass_index for recent candles to check if it was > 27
        let mut had_27 = false;
        for lookback in 1..10.min(n - 25) {
            let lb_start = if n - lookback > 25 { n - lookback - 25 } else { 0 };
            let lb_end = n - lookback;
            let lb_mass: f64 = (lb_start..lb_end)
                .map(|i| {
                    if ema2[i] < 1e-10 { 1.0 } else { ema1[i] / ema2[i] }
                })
                .sum();
            if lb_mass > 27.0 { had_27 = true; break; }
        }
        had_27
    } else {
        false
    };

    (mass_index, bulge)
}
```

### Zone thresholds

| Zone             | Condition                                          |
|------------------|----------------------------------------------------|
| mass-bulge       | mass-index was > 27.0, now < 26.5 (THE signal)    |
| mass-expanding   | mass-index > 27.0 (range expanding)                |
| mass-contracting | mass-index < 25.0 (compressed, trending steadily)  |

### Comparison pairs

None — the Mass Index is a standalone regime-change detector. It does not
compare meaningfully with other indicators on different scales.

### Predicates

```
(at mass-index mass-bulge)           ;; reversal bulge detected (Dorsey signal)
(at mass-index mass-expanding)       ;; range expanding, reversal may be building
(at mass-index mass-contracting)     ;; range compressed, trend likely continues
```

### Expert profile

Assign to: `"structure"` (range expansion/contraction is structural).

### Implementation notes

- The Mass Index "normal" value is 25.0 (when EMA9 / EMA9(EMA9) = 1.0 for 25
  periods). Values deviate from 25 when range is expanding (single EMA leads
  double EMA higher, pushing ratio > 1) or contracting (ratio < 1).
- The reversal bulge is a STATEFUL signal — it requires crossing above 27 THEN
  falling below 26.5. This needs either:
  (a) A sliding window check over recent mass-index values (shown above), or
  (b) A boolean state variable in ThoughtEncoder that tracks "was_above_27".
  Option (b) is cleaner and matches how existing zone checks work with prev candle.
- The Mass Index does NOT indicate direction of the reversal. It must be combined
  with directional indicators (trend, RSI, MACD) to determine whether the
  reversal is from up-to-down or down-to-up. This is exactly the kind of
  composition the journaler excels at discovering.
- 27.0 and 26.5 are Dorsey's original thresholds. They are well-tested on equities
  and commodities. For 5-minute BTC data, they may need calibration — consider
  logging mass-index values during initial runs to verify the thresholds produce
  reasonable bulge frequency (target: 2-5% of candles, not 0.1% or 30%).

---

## Summary: Atom Count Impact

| Vocabulary         | New Indicators | New Zones | New Comparison Pairs | Total New Atoms |
|--------------------|---------------|-----------|---------------------|-----------------|
| Z-Score            | 3             | 6         | 2                   | 9 + 2 pairs     |
| Autocorrelation    | 2             | 5         | 1                   | 7 + 1 pair      |
| Entropy            | 1             | 4         | 0                   | 5               |
| Donchian Channels  | 5             | 6         | 4                   | 11 + 4 pairs    |
| Ultimate Oscillator| 1             | 5         | 1                   | 6 + 1 pair      |
| Mass Index         | 1             | 3         | 0                   | 4               |
| **TOTAL**          | **13**        | **29**    | **8**               | **42 + 8 pairs**|

Current atom count: ~84. After adding these: ~126 atoms.
Current comparison pairs: ~35. After adding these: ~43 pairs.

This is a ~50% expansion in atom vocabulary but targets entirely different
information dimensions than the existing indicators:

- **Existing vocab** captures: level relationships, trend direction, oscillator zones,
  candlestick structure, calendar effects.
- **New vocab** captures: distributional position (z-score), return persistence regime
  (autocorrelation), predictability regime (entropy), absolute range position
  (Donchian), multi-timeframe momentum agreement (Ultimate Oscillator), range
  expansion dynamics (Mass Index).

The new indicators are nearly orthogonal to existing ones in concept space,
which means they add genuine new information to the thought vector rather than
redundant reformulations of existing facts.

---

## Implementation Order (recommended)

1. **Z-Score** — highest expected impact. Directly addresses the L1 prototype
   blurring problem by providing normalized distance rather than raw proximity.
   When all candles fire `(above close sma20)`, the z-score zone differentiates
   "barely above" from "2 sigma above."

2. **Entropy** — meta-signal that can modulate conviction without adding
   directional noise. If high-entropy candles get lower weight in accumulation,
   discriminant strength should improve.

3. **Donchian Channels** — simple, robust, and complementary to Bollinger Bands.
   Provides absolute-range context that BB's standard-deviation approach misses.

4. **Autocorrelation** — regime classifier. Tells the journaler whether to trust
   momentum or mean-reversion signals. Could resolve second-half accuracy decay
   if regime changes are the cause.

5. **Ultimate Oscillator** — natural complement to RSI. The UO-vs-RSI comparison
   is a signal no other indicator pair provides.

6. **Mass Index** — most specialized. Reversal bulge is rare but high-value.
   Implement last because it requires stateful bulge tracking.
