# Batch 019 — Cross-Domain Vocabulary Analysis

Six non-TA domains evaluated for computable signals from 48 candles of OHLCV.
Honest assessment: what is real, what is metaphor, what is charlatan nonsense.

Input: 48 Candle structs with ts, OHLCV, sma20/50/200, bb_upper/lower,
rsi, macd_line/signal/hist, dmi_plus/minus, adx, atr_r.

Constraint: all computations must produce atoms compatible with the existing
`ThoughtEncoder` pipeline — facts composed via `bind(predicate, bind(a, b))`,
bundled into a thought vector.

---

## 1. Thermodynamics

### The honest assessment

Most "thermodynamic" trading metaphors are pure hand-waving. "The market is
heating up" means nothing computable. But three specific concepts map to
well-defined computations on OHLCV:

**Temperature (kinetic energy analog):** Realized variance over a window IS
a temperature. In statistical mechanics, temperature is proportional to the
mean kinetic energy of particles. Price returns ARE the velocities. The sum
of squared returns over N candles is literally the realized variance, and
it is mathematically identical to the temperature of a particle system where
each return is a velocity sample. This is not metaphor — it is the exact
same equation.

**Entropy (thermodynamic, not Shannon):** The Boltzmann entropy of a
distribution of return magnitudes across energy levels. Distinct from the
Shannon entropy in the quant vocab: Shannon entropy bins returns by
direction and magnitude, while thermodynamic entropy tracks how energy
(variance) is distributed across timescales. A market with all variance
concentrated at one timescale (e.g., a single spike) has low thermodynamic
entropy. A market with variance evenly spread across timescales has high
entropy. This is computable via the power spectrum of returns.

**Phase transitions:** A sudden jump in realized variance relative to its
recent trend. In physics, a phase transition is a discontinuity in the order
parameter. Here the order parameter is realized variance, and a phase
transition is when `var(last_10) / var(prev_10)` exceeds a threshold. This
has been tested: Cont (2001) showed return variance exhibits clustering and
regime-switching that resembles critical phenomena. Bouchaud and Potters
(2003) explicitly modeled volatility as a thermodynamic system.

**Pressure (volume-weighted price displacement):** Pressure = force / area.
In a price series, volume is the "mass" and return is the "velocity," so
volume * |return| is the "kinetic energy" per candle, and the mean over a
window is the pressure. High pressure = large volume driving large moves.
This is essentially volume-weighted volatility, which is used by quant
desks (VWAP variants use similar logic).

**What is NOT real:** "Market entropy" without a precise definition. "Energy
levels" of support/resistance. "Conservation of momentum in prices" (prices
do NOT conserve momentum — that is physics cargo-culting).

### Has anyone tested this on financial data?

Yes. Extensively:
- Bouchaud & Potters, "Theory of Financial Risk and Derivative Pricing" (2003):
  explicit temperature/variance equivalence for financial systems.
- Cont (2001): volatility clustering as a critical phenomenon.
- Multifractal models (Mandelbrot, Calvet & Fisher): energy cascade from
  large to small timescales, directly testable on OHLCV.
- Entropy-based portfolio optimization (Philippatos & Wilson, 1972) and its
  modern descendants are standard in quant asset allocation.

### Verdict: GENUINE UNDERDOG

Temperature, spectral entropy, and phase transitions are real computables
with published financial applications. Pressure is a weaker signal but
trivially computable. This vocabulary captures information that no standard
TA indicator measures: the distribution of variance across timescales and
abrupt regime changes in that distribution.

### Atoms

```
;; ─── NEW INDICATORS (5) ─────────────────────────────────────────
thermo-temp          ;; realized variance of log returns over 48 candles
                     ;; (sum of squared returns / n) — literally temperature
thermo-temp-short    ;; realized variance over last 12 candles
thermo-temp-ratio    ;; thermo-temp-short / thermo-temp
                     ;; >1.0 = heating up, <1.0 = cooling down
spectral-entropy     ;; entropy of power spectrum of returns (0-1 normalized)
                     ;; high = variance spread across frequencies (disordered)
                     ;; low = variance concentrated at one frequency (ordered)
thermo-pressure      ;; mean(volume * |log_return|) over 48 candles
                     ;; volume-weighted volatility

;; ─── NEW ZONES (8) ──────────────────────────────────────────────
hot                  ;; thermo-temp > 2x median historical temp (high volatility regime)
warm                 ;; thermo-temp > 1.2x median, <= 2x
cool                 ;; thermo-temp > 0.5x median, <= 1.2x
cold                 ;; thermo-temp <= 0.5x median (suppressed volatility)
heating              ;; thermo-temp-ratio > 1.5 (variance accelerating)
cooling              ;; thermo-temp-ratio < 0.67 (variance decelerating)
phase-transition     ;; thermo-temp-ratio > 3.0 OR < 0.33
                     ;; (abrupt variance regime change — the phase transition)
spectral-ordered     ;; spectral-entropy < 0.4 (dominant frequency exists)
```

### Computation from 48-candle window

```rust
fn compute_thermo_temp(candles: &[Candle]) -> f64 {
    // Realized variance = mean of squared log returns = temperature
    let n = candles.len();
    if n < 5 { return 0.0; }
    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();
    returns.iter().map(|r| r * r).sum::<f64>() / returns.len() as f64
}

fn compute_thermo_temp_short(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 13 { return compute_thermo_temp(candles); }
    compute_thermo_temp(&candles[n - 12..])
}

fn compute_spectral_entropy(candles: &[Candle]) -> f64 {
    // Power spectrum via periodogram, then Shannon entropy of normalized spectrum
    let n = candles.len();
    if n < 16 { return 1.0; }
    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();
    let r_n = returns.len();

    // DFT magnitudes (no FFT needed for N=47 — O(N^2) is ~2200 ops)
    let num_freqs = r_n / 2;
    let mut power = Vec::with_capacity(num_freqs);
    for k in 1..=num_freqs {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (t, &r) in returns.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * t as f64 / r_n as f64;
            re += r * angle.cos();
            im += r * angle.sin();
        }
        power.push(re * re + im * im);
    }

    // Normalize to probability distribution
    let total: f64 = power.iter().sum();
    if total < 1e-20 { return 1.0; }
    let max_entropy = (num_freqs as f64).ln();
    if max_entropy < 1e-10 { return 1.0; }

    let entropy: f64 = power.iter()
        .map(|&p| {
            let prob = p / total;
            if prob < 1e-15 { 0.0 } else { -prob * prob.ln() }
        })
        .sum();

    entropy / max_entropy // 0 = one frequency dominates, 1 = white noise
}

fn compute_thermo_pressure(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 2 { return 0.0; }
    let mut sum = 0.0_f64;
    for w in candles.windows(2) {
        let log_ret = (w[1].close / w[0].close).ln().abs();
        sum += w[1].volume * log_ret;
    }
    sum / (n - 1) as f64
}
```

### Comparison pairs

```
("thermo-temp", "thermo-temp-short"),  ;; long vs short variance (redundant with ratio, but
                                       ;; the cross/touch predicates on this pair detect
                                       ;; the exact moment of regime change)
```

### Predicates

```
(at thermo-temp hot)                    ;; high-variance regime
(at thermo-temp cold)                   ;; suppressed variance (breakout coming?)
(at thermo-temp-ratio heating)          ;; variance accelerating
(at thermo-temp-ratio cooling)          ;; variance decelerating
(at thermo-temp-ratio phase-transition) ;; abrupt regime change
(at spectral-entropy spectral-ordered)  ;; dominant frequency — periodic behavior
```

### Expert profile

Assign to: `"structure"` (variance regime is structural context, not directional).

### What this captures that existing vocab misses

ATR and BB-width measure volatility but not its timescale structure. A market
can have the same ATR but very different spectral entropy: one with a dominant
12-candle cycle vs. one with white-noise returns. The temperature ratio
detects regime changes faster than ATR (which has a 14-period built-in lag).
The phase-transition zone is the most novel signal — it fires on the exact
candle where variance regime shifts, which is when most TA indicators fail.

---

## 2. Fluid Dynamics

### The honest assessment

This one is mostly metaphor with exactly two exceptions that are real:

**Reynolds number analog — REAL:** The Reynolds number in fluid dynamics is
the ratio of inertial forces to viscous forces: `Re = velocity * length / viscosity`.
For a price series: velocity = rate of price change, length = number of candles
in the current trend, viscosity = resistance to movement (proportional to
inverse volume or to range/|return| ratio). The resulting dimensionless number
classifies flow as laminar (smooth, trending) or turbulent (chaotic, choppy).
This is exactly the Hurst exponent classification but computed differently and
potentially carrying different information.

The specific computation: `Re = |mean_return| * trend_length / (std_return / sqrt(volume_ratio))`.
When Re is high, the trend is strong relative to noise (laminar). When Re is
low, noise dominates (turbulent). The critical Re in fluid dynamics is ~2300;
the critical trading Re needs calibration but the concept is sound.

**Viscosity (bid-ask friction proxy) — PARTIALLY REAL:** In fluids, viscosity
is resistance to deformation. In a price series, the ratio of price range to
net movement measures "wasted motion" — how much the price moved internally
vs. how far it actually got. High viscosity = large ranges but small net moves
(choppy, range-bound). Low viscosity = range approximately equals net movement
(clean trend). This is `(sum_of_ranges) / |net_price_change|` over a window,
which is the well-known "efficiency ratio" used by Perry Kaufman in his
Adaptive Moving Average (1995). So this is a real signal, though it has a
different name in the TA literature.

**Boundary layers — MOSTLY METAPHOR:** The idea that price has a "boundary
layer" near support/resistance where behavior changes is appealing but not
well-defined enough to compute. You can detect proximity to extremes (which
Donchian channels already do) but the fluid dynamics framing adds no
computational content.

**Laminar vs turbulent flow — REDUNDANT:** This is just another way to say
"trending vs choppy," which ADX and the efficiency ratio already capture.
The Reynolds number analog is the only version that packages it differently
enough to potentially carry new information.

### Has anyone tested this on financial data?

Partially:
- Kaufman's Efficiency Ratio (1995) is the viscosity concept. Widely tested,
  used in adaptive systems.
- The Hurst exponent (closely related to the Reynolds analog) has extensive
  financial literature. Peters (1994) "Fractal Market Hypothesis."
- No one has explicitly computed a Reynolds number for price series, but the
  components are all individually well-studied.

### Verdict: ONE GENUINE SIGNAL, REST IS METAPHOR

The efficiency ratio / viscosity measure is real and well-tested (it IS
the Kaufman Efficiency Ratio). The Reynolds number is a novel repackaging
that might carry additional information through its volume weighting. The
rest (boundary layers, turbulence) is metaphor rebranding of existing
concepts.

Worth implementing: viscosity (efficiency ratio) and the volume-weighted
Reynolds analog. Not worth implementing: anything labeled "boundary layer,"
"laminar flow," or "turbulent flow" — these are just ADX with extra words.

### Atoms

```
;; ─── NEW INDICATORS (3) ─────────────────────────────────────────
flow-viscosity       ;; sum(candle_range) / |net_price_change| over 20 candles
                     ;; Kaufman Efficiency Ratio inverted: high = choppy, low = efficient
                     ;; (inverted because viscosity = resistance to movement)
flow-reynolds        ;; |mean_return| * trend_length * sqrt(volume_ratio) / std_return
                     ;; dimensionless: high = smooth trend, low = turbulent chop
flow-vorticity       ;; rate of change of direction: count of sign changes in returns
                     ;; over last 20 candles, normalized to [0, 1]
                     ;; 1.0 = alternating every candle, 0.0 = same direction throughout

;; ─── NEW ZONES (6) ──────────────────────────────────────────────
laminar              ;; flow-reynolds > 2.0 (smooth trend, little noise)
transitional         ;; 0.8 < flow-reynolds <= 2.0
turbulent            ;; flow-reynolds <= 0.8 (noise dominates)
high-viscosity       ;; flow-viscosity > 5.0 (lots of wasted motion)
low-viscosity        ;; flow-viscosity < 2.0 (efficient price movement)
high-vorticity       ;; flow-vorticity > 0.6 (rapid direction changes)
```

### Computation from 48-candle window

```rust
fn compute_viscosity(candles: &[Candle]) -> f64 {
    // Kaufman Efficiency Ratio (inverted): sum(ranges) / |net move|
    // High = choppy (high resistance), low = trending (low resistance)
    let n = candles.len();
    let period = 20.min(n);
    let start = n - period;
    let slice = &candles[start..n];

    let sum_ranges: f64 = slice.iter()
        .map(|c| (c.high - c.low).max(1e-10))
        .sum();
    let net_move = (slice.last().unwrap().close - slice[0].close).abs();
    if net_move < 1e-10 { return 100.0; } // infinite viscosity = no net movement
    sum_ranges / net_move
}

fn compute_reynolds(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 10 { return 0.0; }

    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln())
        .collect();

    let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_ret = {
        let var: f64 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>()
            / returns.len() as f64;
        var.sqrt()
    };
    if std_ret < 1e-15 { return 0.0; }

    // Trend length: number of candles since last sign change in cumulative return
    let mut trend_len = 0usize;
    let mut cum_sign = if mean_ret >= 0.0 { 1 } else { -1 };
    for &r in returns.iter().rev() {
        let s = if r >= 0.0 { 1 } else { -1 };
        if s == cum_sign { trend_len += 1; } else { break; }
    }
    let trend_len = (trend_len as f64).max(1.0);

    // Volume ratio: recent volume / early volume (momentum of participation)
    let mid = n / 2;
    let vol_early: f64 = candles[..mid].iter().map(|c| c.volume).sum::<f64>() / mid as f64;
    let vol_late: f64 = candles[mid..].iter().map(|c| c.volume).sum::<f64>()
        / (n - mid) as f64;
    let vol_ratio = if vol_early < 1e-10 { 1.0 } else { vol_late / vol_early };

    mean_ret.abs() * trend_len * vol_ratio.sqrt() / std_ret
}

fn compute_vorticity(candles: &[Candle]) -> f64 {
    // Fraction of sign changes in returns over last 20 candles
    let n = candles.len();
    let period = 20.min(n - 1);
    let start = n - period - 1;
    let returns: Vec<f64> = candles[start..n].windows(2)
        .map(|w| w[1].close - w[0].close)
        .collect();
    if returns.len() < 2 { return 0.5; }

    let sign_changes = returns.windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    sign_changes as f64 / (returns.len() - 1) as f64
}
```

### Comparison pairs

```
("flow-reynolds", "adx"),     ;; Reynolds vs ADX — both measure trend quality
                              ;; but computed from completely different inputs.
                              ;; Divergence = the signal.
```

### Predicates

```
(at flow-reynolds laminar)          ;; smooth trend, momentum reliable
(at flow-reynolds turbulent)        ;; choppy, momentum unreliable
(at flow-viscosity high-viscosity)  ;; wasted motion, range-bound
(at flow-viscosity low-viscosity)   ;; efficient movement, trending
(at flow-vorticity high-vorticity)  ;; rapid reversals (whipsaw zone)
(above flow-reynolds adx)           ;; Reynolds says trend but ADX doesn't (or vice versa)
```

### Expert profile

Assign to: `"structure"` (flow regime is structural context).

### What this captures that existing vocab misses

ADX measures trend strength from directional movement. The Reynolds analog
weights by volume participation and trend duration, which ADX does not.
Vorticity (sign-change frequency) is not captured by any existing indicator —
it measures choppiness at the individual-candle level, while ADX smooths over
14 periods. The viscosity measure (efficiency ratio) is well-tested but not
in the current vocab.

---

## 3. Ecology / Biology

### The honest assessment

Most biological metaphors applied to markets are pure marketing ("bull and
bear" is already zoology, and it carries zero computational content). But
two concepts have genuine mathematical substance:

**Lotka-Volterra (predator-prey) dynamics — REAL:** The Lotka-Volterra
equations model oscillating populations: when prey is abundant, predators
grow; when predators peak, prey collapses; predators then starve and prey
recovers. In markets: "buyers" (upward pressure) and "sellers" (downward
pressure) oscillate. The measurable proxy is the ratio of buying volume to
selling volume, approximated by classifying candles as "buyer-dominated"
(close > open, volume weighted by body/range ratio) vs "seller-dominated."

The key insight: in Lotka-Volterra, the RATE OF CHANGE of each population
depends on the OTHER population's size. If we compute rolling "buyer
strength" and "seller strength," the cross-derivative (does increasing buyer
strength predict decreasing seller strength next?) is a Lotka-Volterra test.
If the dynamics fit, we expect oscillation with a measurable period.

This has been tested: Farmer & Joshi (2002) modeled market microstructure as
an ecology of strategies. Bouchaud et al. (2009) showed order flow has
predator-prey dynamics between momentum and mean-reversion agents.

**Carrying capacity — PARTIALLY REAL:** In ecology, carrying capacity is the
maximum population an environment can sustain. In markets, the analog is the
maximum trend extension before mean reversion — how far can price deviate
from equilibrium (e.g., moving average) before it "runs out of food" (buying
power)? This is computable as the historical maximum z-score achieved before
reversal. With 48 candles you can estimate recent carrying capacity.

However, this is essentially a restatement of z-score with a dynamic
threshold, which the z-score vocabulary already provides. The biological
framing adds nothing computational.

**Population dynamics of order flow — METAPHOR:** Without Level 2 order
book data, we cannot observe actual populations of orders. OHLCV gives us
aggregate outcomes, not individual agents. Any "population dynamics" we
compute from OHLCV is really just volume analysis with a biological label.

### Has anyone tested this on financial data?

Yes:
- Farmer & Joshi (2002): explicit ecology-of-strategies model. Strategies
  have birth/death rates. Published in Nature.
- Bouchaud, Farmer, Lillo (2009): "How Markets Slowly Digest Changes in
  Supply and Demand" — predator-prey dynamics in order flow.
- Lux & Marchesi (1999): agent-based model with chartists/fundamentalists
  as competing species. Reproduces stylized facts.
- "Carrying capacity" in the Kelly criterion sense: the market has a maximum
  rate of return extraction before the strategy degrades its own edge.

### Verdict: ONE GENUINE SIGNAL (buyer/seller oscillation), REST IS RELABELING

The buyer/seller strength oscillation can be computed from OHLCV and captures
information not in any standard indicator. The phase of the buyer/seller
cycle at the current candle is a potentially predictive feature. Everything
else is either z-score in disguise or requires data we don't have.

### Atoms

```
;; ─── NEW INDICATORS (4) ─────────────────────────────────────────
buyer-strength       ;; rolling sum of (volume * body_ratio) for bullish candles
                     ;; over last 20 candles, normalized by total volume
seller-strength      ;; same for bearish candles
predator-prey-phase  ;; atan2(d_seller/dt, d_buyer/dt) mapped to 4 phases:
                     ;; growth (buyers rising), peak (sellers rising),
                     ;; decline (buyers falling), trough (sellers falling)
predator-prey-ratio  ;; buyer-strength / seller-strength (log scale)
                     ;; >0 = buyer-dominated, <0 = seller-dominated

;; ─── NEW ZONES (6) ──────────────────────────────────────────────
buyer-dominated      ;; predator-prey-ratio > 0.3 (buyers clearly winning)
seller-dominated     ;; predator-prey-ratio < -0.3 (sellers clearly winning)
eco-balanced         ;; -0.3 <= predator-prey-ratio <= 0.3 (contested)
prey-abundant        ;; phase = growth (buyers rising, sellers stable — easy up)
predator-peak        ;; phase = peak (sellers surging — reversal from up imminent)
prey-scarce          ;; phase = decline (buyers exhausted — down move underway)
```

### Computation from 48-candle window

```rust
fn compute_buyer_seller_strength(candles: &[Candle]) -> (f64, f64) {
    let n = candles.len();
    let period = 20.min(n);
    let start = n - period;

    let mut buyer_strength = 0.0_f64;
    let mut seller_strength = 0.0_f64;
    let mut total_vol = 0.0_f64;

    for c in &candles[start..n] {
        let range = c.high - c.low;
        if range < 1e-10 { continue; }
        let body_ratio = (c.close - c.open).abs() / range; // conviction
        let contribution = c.volume * body_ratio;
        total_vol += c.volume;

        if c.close > c.open {
            buyer_strength += contribution;
        } else {
            seller_strength += contribution;
        }
    }
    if total_vol < 1e-10 { return (0.5, 0.5); }
    (buyer_strength / total_vol, seller_strength / total_vol)
}

fn compute_predator_prey_phase(candles: &[Candle]) -> &'static str {
    // Compute buyer/seller strength at two points to get derivatives
    let n = candles.len();
    if n < 30 { return "eco-balanced"; }

    let (b_now, s_now) = compute_buyer_seller_strength(&candles[n - 20..]);
    let (b_prev, s_prev) = compute_buyer_seller_strength(&candles[n - 30..n - 10]);

    let db = b_now - b_prev; // buyer derivative
    let ds = s_now - s_prev; // seller derivative

    // Four-phase classification (like Lotka-Volterra orbit quadrants):
    if db > 0.0 && ds <= 0.0 { "prey-abundant" }   // buyers rising, sellers stable/falling
    else if db >= 0.0 && ds > 0.0 { "predator-peak" } // both rising, sellers catching up
    else if db < 0.0 && ds >= 0.0 { "prey-scarce" }   // buyers falling, sellers still strong
    else { "prey-recovery" }                            // both falling, cycle resetting
}

fn compute_predator_prey_ratio(candles: &[Candle]) -> f64 {
    let (b, s) = compute_buyer_seller_strength(candles);
    if s < 1e-10 { return 3.0; }
    if b < 1e-10 { return -3.0; }
    (b / s).ln().clamp(-3.0, 3.0)
}
```

### Comparison pairs

```
("buyer-strength", "seller-strength"),   ;; direct strength comparison
```

### Predicates

```
(at predator-prey-ratio buyer-dominated)    ;; buyers in control
(at predator-prey-ratio seller-dominated)   ;; sellers in control
(at predator-prey-phase prey-abundant)      ;; Lotka-Volterra: up phase
(at predator-prey-phase predator-peak)      ;; Lotka-Volterra: top reversal phase
(at predator-prey-phase prey-scarce)        ;; Lotka-Volterra: down phase
(above buyer-strength seller-strength)      ;; buyers stronger
(crosses-below buyer-strength seller-strength) ;; sellers taking over
```

### Expert profile

Assign to: `"volume"` (this is fundamentally a volume-partitioning signal).

### What this captures that existing vocab misses

The existing `eval_volume_confirmation` checks whether volume confirms price
direction. The predator-prey vocabulary goes further: it tracks the RATE OF
CHANGE of buyer vs seller strength and classifies the cycle phase. This is
information about where we are in the buyer/seller oscillation, not just
whether the current candle's volume agrees with its direction. The phase
classification is genuinely novel — I have not seen it in standard TA
indicator libraries.

---

## 4. Music Theory

### The honest assessment

Music theory applied to markets sounds like astrology. It is mostly astrology.
But two concepts survive the honesty filter:

**Tempo (rhythm regularity) — REAL:** In music, tempo is the rate of beats
per unit time, and tempo changes signal structural transitions. In a price
series, the "beat" is a significant move (return exceeding a threshold). The
inter-beat interval (time between significant moves) has a measurable
distribution. When the tempo accelerates (intervals shorten), the market is
"speeding up" — more significant moves per unit time. When it decelerates,
the market is settling. This is computable and carries information not in
any standard indicator: it measures the TIMING of significant moves, not
their magnitude.

This is related to the concept of "activity rate" in point process theory.
Hawkes processes (self-exciting point processes) have been extensively used
in financial modeling. The inter-event time distribution is a key feature
of Hawkes processes.

**Syncopation (pattern violation) — PARTIALLY REAL:** Syncopation is emphasis
on the "wrong" beat — the unexpected. In a price series, if you establish
a regular pattern (e.g., alternating up-down-up-down candles) and then it
breaks (up-up), that's syncopation. This is computable as the surprise
relative to a simple n-gram model of candle directions. High surprise =
high syncopation = the pattern broke. Low surprise = the pattern continues.

This is essentially conditional entropy, which IS a real signal. But it is
very close to what the autocorrelation and entropy vocabularies already
capture. The music framing does not add much beyond relabeling.

**Harmonic intervals between price levels — CHARLATAN:** The idea that price
levels relate to each other by musical ratios (octaves, fifths, etc.) is
Fibonacci-level numerology. There is no physical reason for prices to
respect musical intervals. No published evidence supports this.

**Chord detection (co-moving indicators) — METAPHOR:** The idea that when
multiple indicators "harmonize" (agree) it's like a chord is just another
way to say "indicator confluence." The existing comparison pairs already
detect this. Calling it a "chord" adds no computational content.

### Has anyone tested this on financial data?

Tempo/event rate: yes, extensively through Hawkes process literature:
- Bacry, Mastromatteo, Muzy (2015): "Hawkes processes in finance" — reviews
  hundreds of papers on self-exciting point processes for financial events.
- Hardiman, Bercot, Bouchaud (2013): calibrated Hawkes processes on market
  microstructure.

Syncopation/surprise: partially, through entropy rate estimation:
- Kontoyiannis et al. (1998): entropy rate estimation, applied to various
  sequences including financial.

Harmonics/musical intervals: no legitimate published research supports this.

### Verdict: ONE GENUINE SIGNAL (tempo/event rate), ONE MARGINAL (syncopation), REST IS NONSENSE

### Atoms

```
;; ─── NEW INDICATORS (3) ─────────────────────────────────────────
beat-tempo           ;; number of "significant" moves (|return| > 1.5 * median_abs_return)
                     ;; in last 20 candles, divided by 20 (event rate)
beat-acceleration    ;; tempo_last_10 - tempo_first_10 (speeding up or slowing down)
syncopation          ;; fraction of last 20 candle-direction transitions that violate
                     ;; the most common bigram pattern

;; ─── NEW ZONES (5) ──────────────────────────────────────────────
allegro              ;; beat-tempo > 0.5 (more than half the candles are "significant")
andante              ;; 0.25 < beat-tempo <= 0.5 (moderate activity)
adagio               ;; beat-tempo <= 0.25 (quiet, few significant moves)
accelerando          ;; beat-acceleration > 0.15 (tempo increasing)
ritardando           ;; beat-acceleration < -0.15 (tempo decreasing)
```

### Computation from 48-candle window

```rust
fn compute_beat_tempo(candles: &[Candle], period: usize) -> f64 {
    let n = candles.len();
    let start = if n > period + 1 { n - period - 1 } else { 0 };
    let slice = &candles[start..n];
    if slice.len() < 3 { return 0.0; }

    let returns: Vec<f64> = slice.windows(2)
        .map(|w| (w[1].close / w[0].close).ln().abs())
        .collect();

    // Threshold: 1.5x median absolute return
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = median * 1.5;

    let beats = returns.iter().filter(|&&r| r > threshold).count();
    beats as f64 / returns.len() as f64
}

fn compute_beat_acceleration(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 22 { return 0.0; }
    let tempo_recent = compute_beat_tempo(&candles[n - 11..], 10);
    let tempo_earlier = compute_beat_tempo(&candles[n - 21..n - 10], 10);
    tempo_recent - tempo_earlier
}

fn compute_syncopation(candles: &[Candle]) -> f64 {
    let n = candles.len();
    let period = 20.min(n - 1);
    let start = n - period - 1;

    // Direction sequence: +1 (up), -1 (down)
    let dirs: Vec<i8> = candles[start..n].windows(2)
        .map(|w| if w[1].close >= w[0].close { 1 } else { -1 })
        .collect();

    if dirs.len() < 3 { return 0.0; }

    // Count bigrams: (up,up), (up,down), (down,up), (down,down)
    let mut counts = [[0u32; 2]; 2]; // [prev_dir][curr_dir]
    for w in dirs.windows(2) {
        let prev = if w[0] > 0 { 1 } else { 0 };
        let curr = if w[1] > 0 { 1 } else { 0 };
        counts[prev][curr] += 1;
    }

    // For each bigram, probability of the actual next direction
    // Syncopation = fraction of transitions where the minority outcome occurred
    let mut surprises = 0u32;
    let mut total = 0u32;
    for w in dirs.windows(2) {
        let prev = if w[0] > 0 { 1 } else { 0 };
        let curr = if w[1] > 0 { 1 } else { 0 };
        let other = 1 - curr;
        if counts[prev][curr] < counts[prev][other] {
            surprises += 1; // this was the less common transition
        }
        total += 1;
    }
    if total == 0 { return 0.0; }
    surprises as f64 / total as f64
}
```

### Comparison pairs

None — these are standalone regime descriptors.

### Predicates

```
(at beat-tempo allegro)              ;; high event rate (lots of significant moves)
(at beat-tempo adagio)               ;; low event rate (quiet market)
(at beat-acceleration accelerando)   ;; tempo increasing (market waking up)
(at beat-acceleration ritardando)    ;; tempo decreasing (market settling)
```

### Expert profile

Assign to: `"narrative"` (tempo is about the temporal pattern, not the level).

### What this captures that existing vocab misses

No existing indicator measures the RATE of significant events. ATR measures
the average magnitude of ranges. The beat tempo measures how FREQUENTLY
ranges exceed a threshold. A market can have constant ATR but variable
tempo: steady moderate moves vs. alternating between quiet candles and large
candles. The acceleration signal detects the transition — the market "waking
up" before ATR catches it.

---

## 5. Linguistics

### The honest assessment

Applying linguistics to price series is surprisingly legitimate because the
core linguistic tools (n-gram frequency analysis, entropy rate, Zipf's law)
are domain-agnostic information theory applied to symbol sequences. A price
series IS a symbol sequence if you discretize the candles.

**Entropy rate (conditional entropy) — REAL:** The entropy rate measures how
unpredictable the NEXT symbol is given the PREVIOUS k symbols. For a candle
sequence discretized into categories (big-up, small-up, doji, small-down,
big-down), the entropy rate answers: "given what just happened, how uncertain
is the next candle?" This is strictly more informative than the Shannon
entropy of returns (which ignores ordering) because it captures sequential
dependence.

This is the key difference from the entropy in the quant vocabulary:
return-entropy measures the distribution of individual returns; the
linguistic entropy rate measures the CONDITIONAL distribution — how
predictable is the SEQUENCE.

**Bigram/trigram frequencies — REAL:** This is literally what candlestick
pattern recognition does, but without naming the patterns. Instead of
defining "bullish engulfing" as a specific two-candle pattern, you compute
the frequency of ALL two-candle transitions. The most interesting feature
is which bigrams are OVER-represented (relative to independence) and which
are UNDER-represented. Over-representation = the market has a habit.
Under-representation = the market avoids this transition. Both are
potentially predictive.

**Zipf's law — PARTIALLY REAL:** Zipf's law says the k-th most common
word in a language has frequency proportional to 1/k. If candle patterns
follow Zipf's law, it implies a specific kind of structure (power-law
distribution of pattern frequencies). Deviations from Zipf might signal
regime change. This is computable but I am skeptical it carries signal
beyond what entropy rate already captures. Testing needed.

**Bigram surprise of current candle — REAL and NOVEL:** Given the previous
candle type, what was the probability of the current candle type under the
observed bigram distribution? Low probability = high surprise = the market
just did something unusual. This is the pointwise mutual information (PMI)
of the current bigram, and it is a per-candle signal that no standard
indicator computes.

### Has anyone tested this on financial data?

Yes:
- Bandt & Pompe (2002): permutation entropy applied to time series,
  including financial data. Widely cited (5000+ citations).
- Zunino et al. (2010): permutation entropy distinguishes stock markets
  by "efficiency" (entropy rate near maximum = efficient market).
- Hou et al. (2017): n-gram models for financial time series classification.
- Cont (2001): stylized facts of return sequences, including serial
  dependence that bigram models capture.

### Verdict: GENUINE UNDERDOG — multiple computable signals carrying novel information

### Atoms

```
;; ─── NEW INDICATORS (4) ─────────────────────────────────────────
entropy-rate         ;; conditional entropy of candle-type sequence (0-1 normalized)
                     ;; given prev candle type, how uncertain is next candle?
                     ;; lower = more predictable = more exploitable
bigram-surprise      ;; -log2(P(current_type | prev_type)) for the most recent transition
                     ;; high = unusual transition just occurred
permutation-entropy  ;; Bandt-Pompe permutation entropy, order 3 (ordinal patterns)
                     ;; lower = more regular sequential structure
zipf-deviation       ;; deviation from Zipf's law of candle-type frequencies
                     ;; high = unusual frequency distribution

;; ─── NEW ZONES (5) ──────────────────────────────────────────────
predictable          ;; entropy-rate < 0.6 (sequence has structure)
unpredictable        ;; entropy-rate > 0.85 (near-random sequence)
surprised            ;; bigram-surprise > 3.0 bits (very unusual transition)
permutation-regular  ;; permutation-entropy < 0.7 (ordinal patterns clustered)
permutation-random   ;; permutation-entropy > 0.9 (ordinal patterns near-uniform)
```

### Computation from 48-candle window

```rust
/// Discretize a candle into one of 5 types based on body and range
fn candle_type(c: &Candle, atr: f64) -> u8 {
    let body = c.close - c.open; // positive = bullish, negative = bearish
    let range = c.high - c.low;
    let atr = atr.max(1e-10);
    let body_atr = body / atr;

    // 5 types: big-down(0), small-down(1), doji(2), small-up(3), big-up(4)
    if body_atr < -0.5 { 0 }       // big bearish
    else if body_atr < -0.1 { 1 }  // small bearish
    else if body_atr < 0.1 { 2 }   // doji
    else if body_atr < 0.5 { 3 }   // small bullish
    else { 4 }                      // big bullish
}

fn compute_entropy_rate(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 10 { return 1.0; }

    // Use ATR from the last candle as normalization
    let atr = candles.last().unwrap().atr_r;
    let types: Vec<u8> = candles.iter().map(|c| candle_type(c, atr)).collect();

    // Bigram counts: P(type_t | type_{t-1})
    let num_types = 5u8;
    let mut bigram_counts = vec![vec![0u32; num_types as usize]; num_types as usize];
    let mut unigram_counts = vec![0u32; num_types as usize];

    for w in types.windows(2) {
        bigram_counts[w[0] as usize][w[1] as usize] += 1;
        unigram_counts[w[0] as usize] += 1;
    }

    // Conditional entropy: H(X_t | X_{t-1}) = -sum P(x,y) log P(y|x)
    let total = (types.len() - 1) as f64;
    let mut h = 0.0_f64;
    for prev in 0..num_types as usize {
        if unigram_counts[prev] == 0 { continue; }
        for curr in 0..num_types as usize {
            if bigram_counts[prev][curr] == 0 { continue; }
            let p_joint = bigram_counts[prev][curr] as f64 / total;
            let p_cond = bigram_counts[prev][curr] as f64 / unigram_counts[prev] as f64;
            h -= p_joint * p_cond.ln();
        }
    }

    // Normalize by max possible entropy (ln(5))
    let max_h = (num_types as f64).ln();
    if max_h < 1e-10 { return 1.0; }
    (h / max_h).clamp(0.0, 1.0)
}

fn compute_bigram_surprise(candles: &[Candle]) -> f64 {
    let n = candles.len();
    if n < 10 { return 0.0; }

    let atr = candles.last().unwrap().atr_r;
    let types: Vec<u8> = candles.iter().map(|c| candle_type(c, atr)).collect();

    let prev_type = types[types.len() - 2] as usize;
    let curr_type = types[types.len() - 1] as usize;

    // Count how often prev_type was followed by each type
    let mut counts = vec![0u32; 5];
    let mut total = 0u32;
    for w in types.windows(2) {
        if w[0] as usize == prev_type {
            counts[w[1] as usize] += 1;
            total += 1;
        }
    }
    if total == 0 { return 0.0; }

    let p = counts[curr_type] as f64 / total as f64;
    if p < 1e-10 { return 10.0; } // never seen = maximum surprise
    -p.log2() // surprise in bits
}

fn compute_permutation_entropy(candles: &[Candle]) -> f64 {
    // Bandt-Pompe permutation entropy, order 3
    // Count ordinal patterns in consecutive triples of close prices
    let n = candles.len();
    if n < 10 { return 1.0; }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();

    // Order 3: 6 possible permutation patterns (3! = 6)
    let mut pattern_counts = [0u32; 6];
    for w in closes.windows(3) {
        // Encode the ordinal pattern: which of 6 orderings?
        let pat = if w[0] <= w[1] && w[1] <= w[2] { 0 }       // ascending
            else if w[0] <= w[2] && w[2] <= w[1] { 1 }         // peak at middle
            else if w[1] <= w[0] && w[0] <= w[2] { 2 }         // valley start
            else if w[2] <= w[0] && w[0] <= w[1] { 3 }         // (210 rev)
            else if w[1] <= w[2] && w[2] <= w[0] { 4 }         // (120 rev)
            else { 5 };                                          // descending
        pattern_counts[pat] += 1;
    }

    let total = pattern_counts.iter().sum::<u32>() as f64;
    if total < 1.0 { return 1.0; }
    let max_h = (6.0_f64).ln();

    let h: f64 = pattern_counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();

    (h / max_h).clamp(0.0, 1.0)
}
```

### Comparison pairs

```
("entropy-rate", "return-entropy"),    ;; conditional vs unconditional entropy
                                       ;; divergence = sequential structure exists
                                       ;; beyond what marginal distribution shows
```

### Predicates

```
(at entropy-rate predictable)                ;; candle sequence has exploitable structure
(at entropy-rate unpredictable)              ;; candle sequence is near-random
(at bigram-surprise surprised)               ;; unusual transition just happened
(at permutation-entropy permutation-regular) ;; ordinal patterns clustered (structure)
(below entropy-rate return-entropy)          ;; conditional < unconditional = sequential dependence!
```

### Expert profile

Assign to: `"narrative"` (this is about sequential structure in the candle narrative).

### What this captures that existing vocab misses

The return-entropy from the quant vocabulary measures whether returns are
uniformly distributed. The entropy rate measures whether the SEQUENCE of
returns has structure beyond what the distribution implies. Example: a coin
that goes HTHTHTHT has maximum entropy (equal H and T) but minimum entropy
rate (perfectly predictable given the previous flip). The gap between
`return-entropy` and `entropy-rate` IS the amount of sequential structure —
this is genuinely novel information for the thought vector.

The bigram surprise is a per-candle signal: "was this transition unusual?"
No existing indicator answers this question. It fires on pattern breaks
regardless of the direction of the break.

---

## 6. Seismology

### The honest assessment

Seismology has the strongest theoretical foundation of any domain on this
list, because earthquakes and price crashes share the same underlying
mathematics: power-law distributions of event sizes, temporal clustering
of events, and aftershock decay following empirical laws.

**Gutenberg-Richter law (frequency-magnitude relationship) — REAL:**
The Gutenberg-Richter law states that the number of earthquakes with
magnitude >= M is proportional to 10^(-bM), where b is a constant near 1.
In price series, replace "earthquake" with "return exceeding threshold" and
"magnitude" with |log return|. If the distribution follows a power law, then
b < 1 means heavy tails (more large moves than expected), and b > 1 means
light tails (fewer large moves). Changes in b indicate regime change: the
b-value drops before large earthquakes (and, potentially, before large
price moves).

This is computable from 48 candles and has been tested: Kaizoji (2006)
showed return magnitudes follow Gutenberg-Richter scaling. The b-value
estimated from a 48-candle window tells you whether the tail risk is
elevated.

**Omori's law (aftershock decay) — REAL:** After a mainshock, aftershock
frequency decays as t^(-p), where p is near 1. In price series, after a
large move, the rate of subsequent large moves decays following the same
power law. This means the probability of another large move is highest
immediately after the first one and decays predictably.

The computable signal: given the time since the last "mainshock" (large
return), what is the expected rate of aftershocks (large returns) right now?
If the actual rate exceeds the Omori prediction, the market is more active
than expected (something new is happening). If below, the aftershock
sequence is dying out normally.

**Foreshock detection — PARTIALLY REAL:** In seismology, some large
earthquakes are preceded by foreshock sequences — small events that
accelerate before the mainshock. In price series, an increasing rate of
moderate moves before a large move is a measurable pattern. However, the
predictive power of foreshock detection in actual seismology is debated
(most seismologists say foreshocks can only be identified after the
mainshock). In price series, it is equivalent to detecting increasing
"beat tempo" (from the music vocabulary), so it is real but redundant.

### Has anyone tested this on financial data?

Yes, extensively:
- Kaizoji (2006): power-law distribution of absolute returns mirrors
  Gutenberg-Richter.
- Lillo & Mantegna (2003): Omori-law aftershock decay in financial volatility
  after large events.
- Petersen et al. (2010): "Market Dynamics Immediately Before and After
  Financial Shocks" — explicit Omori law fit to S&P 500 after crashes.
- Sornette & Johansen (2001): log-periodic power law before market crashes
  (the "financial earthquake" model). Controversial but published in mainstream
  physics and finance journals.
- Weber et al. (2007): "Relation between volatility correlations in financial
  markets and Omori processes occurring on all scales" — published in
  Physical Review E.

### Verdict: GENUINE UNDERDOG — the strongest theoretical foundation on this list

Both the Gutenberg-Richter b-value and the Omori aftershock rate are
computable from 48 candles, have published evidence on financial data, and
capture information that no standard TA indicator measures. The b-value
measures tail risk dynamics. The Omori residual measures whether post-shock
decay is proceeding normally or abnormally. Both are novel signals.

### Atoms

```
;; ─── NEW INDICATORS (4) ─────────────────────────────────────────
gr-bvalue            ;; Gutenberg-Richter b-value estimated from return magnitudes
                     ;; lower = heavier tails = more large moves expected
mainshock-age        ;; candles since last "mainshock" (|return| > 3 * median_abs_return)
omori-expected       ;; expected aftershock rate at current mainshock-age
                     ;; from fitted Omori decay: rate = K / (mainshock-age + c)^p
omori-residual       ;; actual_rate - omori-expected
                     ;; positive = more activity than expected (new event?)
                     ;; negative = normal decay (aftershock sequence ending)

;; ─── NEW ZONES (7) ──────────────────────────────────────────────
heavy-tail           ;; gr-bvalue < 0.8 (elevated tail risk)
normal-tail          ;; 0.8 <= gr-bvalue <= 1.3
light-tail           ;; gr-bvalue > 1.3 (suppressed tail risk)
mainshock-recent     ;; mainshock-age < 5 (within immediate aftershock zone)
mainshock-decaying   ;; 5 <= mainshock-age < 20 (aftershock sequence active)
mainshock-quiet      ;; mainshock-age >= 20 (aftershock sequence over)
omori-anomalous      ;; omori-residual > 1.0 (more activity than Omori predicts —
                     ;; either a new mainshock is developing or the sequence is
                     ;; not a simple aftershock pattern)
```

### Computation from 48-candle window

```rust
fn compute_gr_bvalue(candles: &[Candle]) -> f64 {
    // Gutenberg-Richter b-value via maximum likelihood estimation
    // For magnitudes M following P(M >= m) ~ 10^(-bm):
    //   b_MLE = (1 / (mean(M) - M_min)) * log10(e)
    // where M = log10(|return|) and M_min is the completeness threshold
    let n = candles.len();
    if n < 10 { return 1.0; }

    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln().abs())
        .collect();

    // Filter out near-zero returns (below completeness threshold)
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m_min = sorted[sorted.len() / 4]; // 25th percentile as completeness threshold
    if m_min < 1e-10 { return 1.0; }

    let magnitudes: Vec<f64> = returns.iter()
        .filter(|&&r| r >= m_min)
        .map(|&r| r.log10())
        .collect();

    if magnitudes.len() < 5 { return 1.0; }

    let m_min_log = m_min.log10();
    let mean_m: f64 = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
    let denom = mean_m - m_min_log;
    if denom < 1e-10 { return 1.0; }

    // b = log10(e) / (mean(M) - M_min)
    (std::f64::consts::E.log10()) / denom
}

fn compute_mainshock_age(candles: &[Candle]) -> usize {
    let n = candles.len();
    if n < 5 { return n; }

    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln().abs())
        .collect();

    // Threshold: 3x median absolute return
    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = median * 3.0;

    // Find most recent mainshock (counting from the end)
    for (i, &r) in returns.iter().rev().enumerate() {
        if r > threshold {
            return i; // age in candles since mainshock
        }
    }
    returns.len() // no mainshock in window
}

fn compute_omori_residual(candles: &[Candle]) -> f64 {
    // Fit Omori's law: aftershock rate = K / (t + c)^p
    // Simplified: use p=1, c=1 (standard values), estimate K from data
    let n = candles.len();
    if n < 20 { return 0.0; }

    let returns: Vec<f64> = candles.windows(2)
        .map(|w| (w[1].close / w[0].close).ln().abs())
        .collect();

    let mut sorted = returns.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mainshock_thresh = median * 3.0;
    let aftershock_thresh = median * 1.5;

    // Find mainshock
    let mut mainshock_idx: Option<usize> = None;
    for (i, &r) in returns.iter().enumerate().rev() {
        if r > mainshock_thresh {
            mainshock_idx = Some(i);
            break;
        }
    }
    let ms_idx = match mainshock_idx {
        Some(i) if i + 5 < returns.len() => i,
        _ => return 0.0, // no mainshock or too recent
    };

    // Count aftershocks in bins after mainshock
    let after = &returns[ms_idx + 1..];
    if after.len() < 5 { return 0.0; }

    // Split post-mainshock into early (first half) and late (second half)
    let mid = after.len() / 2;
    let early_rate = after[..mid].iter().filter(|&&r| r > aftershock_thresh).count() as f64
        / mid as f64;
    let late_rate = after[mid..].iter().filter(|&&r| r > aftershock_thresh).count() as f64
        / (after.len() - mid) as f64;

    // Omori predicts late_rate < early_rate (decay)
    // Omori expected late rate: early_rate * (early_time / late_time)^p
    // With p=1: expected = early_rate * mid / (after.len() - mid)
    let expected_late = early_rate * mid as f64 / (after.len() - mid) as f64;

    // Residual: actual - expected. Positive = more active than expected.
    late_rate - expected_late
}
```

### Comparison pairs

```
("gr-bvalue", "adx"),   ;; tail risk vs trend strength — divergence is interesting:
                         ;; heavy tails + weak trend = choppy and dangerous
                         ;; heavy tails + strong trend = volatile but directional
```

### Predicates

```
(at gr-bvalue heavy-tail)                  ;; elevated tail risk
(at gr-bvalue light-tail)                  ;; suppressed tail risk (calm, but beware)
(at mainshock-age mainshock-recent)        ;; just had a large move, aftershocks expected
(at mainshock-age mainshock-quiet)         ;; aftershock sequence over (clean slate)
(at omori-residual omori-anomalous)        ;; more activity than decay model predicts
```

### Expert profile

Assign to: `"structure"` (seismic regime is structural context about event dynamics).

### What this captures that existing vocab misses

The Gutenberg-Richter b-value is entirely novel. No existing indicator
measures whether the DISTRIBUTION of return sizes has changed. ATR measures
the mean range. The b-value measures the slope of the cumulative distribution —
whether there are disproportionately many large returns relative to small ones.
A dropping b-value before a crash is the financial equivalent of foreshock
b-value anomaly in seismology.

The Omori aftershock model provides a BASELINE EXPECTATION for post-shock
activity. Without it, every large move after a spike is treated equally. With
it, you can distinguish "expected aftershock" (normal decay, ignore) from
"new event" (above Omori prediction, pay attention). This is information no
other indicator provides.

---

## Summary: Verdict Table

| Domain | Computable signals | Metaphor only | Charlatan |
|--------|-------------------|---------------|-----------|
| Thermodynamics | Temperature (realized variance), spectral entropy, phase transitions, pressure | "Energy levels" | "Conservation of momentum" |
| Fluid Dynamics | Viscosity (efficiency ratio), Reynolds analog, vorticity | Boundary layers | Laminar/turbulent as labels |
| Ecology | Buyer/seller oscillation phase, predator-prey ratio | Carrying capacity | "Population of orders" without L2 data |
| Music Theory | Beat tempo, beat acceleration | Syncopation (≈ conditional entropy) | Harmonic price intervals |
| Linguistics | Entropy rate, bigram surprise, permutation entropy | Zipf deviation | — |
| Seismology | GR b-value, Omori aftershock residual, mainshock age | Foreshock detection (≈ tempo) | — |

## Atom Count Impact

| Vocabulary     | New Indicators | New Zones | New Comparison Pairs | Total New Atoms |
|----------------|---------------|-----------|---------------------|-----------------|
| Thermodynamics | 5             | 8         | 1                   | 13 + 1 pair     |
| Fluid Dynamics | 3             | 6         | 1                   | 9 + 1 pair      |
| Ecology        | 4             | 6         | 1                   | 10 + 1 pair     |
| Music Theory   | 3             | 5         | 0                   | 8               |
| Linguistics    | 4             | 5         | 1                   | 9 + 1 pair      |
| Seismology     | 4             | 7         | 1                   | 11 + 1 pair     |
| **TOTAL**      | **23**        | **37**    | **5**               | **60 + 5 pairs**|

Combined with the quant vocabulary (42 + 8 pairs), this brings the full
batch 019 expansion to **102 new atoms + 13 new comparison pairs**.

## Implementation Priority

1. **Seismology** — strongest theoretical foundation, most novel signals,
   well-tested on financial data. The GR b-value and Omori residual carry
   information that literally no TA indicator measures.

2. **Linguistics** — entropy rate and bigram surprise are genuinely novel
   per-candle signals. The `entropy-rate < return-entropy` comparison
   directly detects exploitable sequential structure.

3. **Thermodynamics** — spectral entropy and phase transitions are strong
   signals. Temperature is partially redundant with ATR but the ratio and
   spectral decomposition are not.

4. **Ecology** — buyer/seller oscillation phase is genuinely novel. The
   Lotka-Volterra phase classification has no equivalent in the existing vocab.

5. **Fluid Dynamics** — viscosity (efficiency ratio) is well-tested but
   partially redundant with ADX. The Reynolds analog adds volume weighting
   which is novel. Lower priority because one of three atoms is a known
   indicator under a different name.

6. **Music Theory** — beat tempo is real but similar to what seismology's
   mainshock-age and linguistics' bigram-surprise capture from different
   angles. Implement last; may be redundant with seismology.

## Key Orthogonality Claims

The following pairs of new signals measure **provably different** information:

- **return-entropy vs entropy-rate**: marginal vs conditional distribution.
  The gap between them IS the mutual information between consecutive candles.
- **GR b-value vs ATR**: slope of size distribution vs mean of size distribution.
  Same ATR can have different b-values.
- **spectral-entropy vs return-entropy**: frequency-domain vs time-domain
  entropy. Same return distribution can have different spectral structure.
- **predator-prey-phase vs volume-confirmation**: oscillation phase vs
  single-candle agreement. Phase captures multi-candle buyer/seller dynamics.
- **bigram-surprise vs any momentum indicator**: surprise is about pattern
  violation, not about direction or magnitude.
- **omori-residual vs ATR**: Omori gives a BASELINE for post-shock activity.
  ATR just measures the level. The residual is the deviation from the
  expected decay, not the level itself.
