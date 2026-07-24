/* Module 7 — monitoring core, JavaScript port.
 *
 * A faithful re-implementation of monitoring/monitoring_core.py so the browser demo can
 * monitor a live scored stream with exactly the maths used in the report: PSI over the
 * deciles of the reference, the Kolmogorov-Smirnov *statistic* (effect size, not p-value),
 * rolling precision / recall / PR-AUC, and the DEFAULT_TRIGGERS retrain rules.
 *
 * Parity with the Python is asserted by tests/test_web_monitoring_parity.py, which feeds
 * identical arrays to both implementations. Keep the two files in step.
 *
 * Also provides makeStream(): a PaySim-shaped transaction generator, so the stream the demo
 * monitors is scored by the real deployed model rather than by a stand-in score.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.FraudMon = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // The 8 features the deployed model consumes (matches monitoring_core.DEPLOY_FEATURES).
  const DEPLOY_FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "type_TRANSFER",
  ];

  const DEFAULT_TRIGGERS = {
    psi_significant: 0.20,   // any served feature with PSI above this
    n_features_drifted: 2,   // at least this many features flagged drifted
    recall_floor: 0.90,      // rolling recall must not dip below this
    precision_floor: 0.20,   // rolling precision must not dip below this (imbalance-aware)
    pr_auc_drop: 0.10,       // rolling PR-AUC must not fall this far below the reference
  };

  // ------------------------------------------------------------------ helpers --
  const asc = (a, b) => a - b;
  const dropNaN = (xs) => Array.prototype.filter.call(xs, (v) => !Number.isNaN(v));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  /** Number of entries <= v in a sorted array (numpy searchsorted side="right"). */
  function upperBound(sorted, v) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  function uniqueSorted(xs) {
    const s = Array.from(xs).sort(asc);
    const out = [];
    for (let i = 0; i < s.length; i++) if (i === 0 || s[i] !== s[i - 1]) out.push(s[i]);
    return out;
  }

  /** np.quantile(..., method="linear") on an already-sorted array. */
  function quantileSorted(sorted, q) {
    const n = sorted.length;
    if (n === 0) return NaN;
    if (n === 1) return sorted[0];
    const pos = q * (n - 1);
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
  }

  /** np.histogram(values, bins=edges): half-open [e_i, e_i+1), last bin closed. */
  function histogram(values, edges) {
    const counts = new Array(edges.length - 1).fill(0);
    for (const v of values) {
      if (Number.isNaN(v)) continue;
      let i = upperBound(edges, v) - 1;
      if (i < 0) continue;                       // below the first edge
      if (i >= counts.length) i = counts.length - 1;  // v == last edge -> closed bin
      counts[i] += 1;
    }
    return counts;
  }

  // -------------------------------------------------------------- data drift --
  /**
   * Population Stability Index between a reference and a current sample of one feature.
   * Bins are the deciles of the reference. PSI < 0.1 = no material shift,
   * 0.1-0.2 = moderate, > 0.2 = significant. Low-cardinality / binary features fall
   * back to a category-frequency comparison.
   */
  function populationStabilityIndex(expected, actual, bins) {
    bins = bins || 10;
    const e0 = dropNaN(expected), a0 = dropNaN(actual);
    if (e0.length === 0 || a0.length === 0) return NaN;

    let e, a;
    const uniqExpected = uniqueSorted(e0);
    if (uniqExpected.length <= bins) {
      // genuinely low-cardinality / binary feature
      const cats = uniqueSorted(e0.concat(a0));
      e = cats.map((c) => e0.filter((v) => v === c).length / e0.length);
      a = cats.map((c) => a0.filter((v) => v === c).length / a0.length);
    } else {
      const sorted = e0.slice().sort(asc);
      const qs = [];
      for (let i = 0; i <= bins; i++) qs.push(quantileSorted(sorted, i / bins));
      const edges = uniqueSorted(qs);
      if (edges.length < 2) return 0.0;  // mass collapsed to one value -> no measurable shift
      edges[0] = -Infinity;
      edges[edges.length - 1] = Infinity;
      e = histogram(e0, edges).map((c) => c / e0.length);
      a = histogram(a0, edges).map((c) => c / a0.length);
    }

    const eps = 1e-6;
    let psi = 0;
    for (let i = 0; i < e.length; i++) {
      const ei = Math.max(e[i], eps), ai = Math.max(a[i], eps);
      psi += (ai - ei) * Math.log(ai / ei);
    }
    return psi;
  }

  /** Two-sample Kolmogorov-Smirnov statistic (scipy.stats.ks_2samp().statistic). */
  function ksStat(a, b) {
    const x = dropNaN(a).slice().sort(asc);
    const y = dropNaN(b).slice().sort(asc);
    if (!x.length || !y.length) return NaN;
    let d = 0;
    for (const arr of [x, y]) {
      for (const v of arr) {
        const diff = Math.abs(upperBound(x, v) / x.length - upperBound(y, v) / y.length);
        if (diff > d) d = diff;
      }
    }
    return d;
  }

  /**
   * Per-feature drift between a reference window and a current window.
   * Flagged when PSI > psiThreshold OR the KS statistic > ksStatThreshold.
   *
   * The flag keys on the KS *statistic*, not its p-value: at production sample sizes the
   * p-value is < 0.01 for any negligible difference, so a p-value rule would flag every
   * feature. PSI and the statistic are effect sizes and stay meaningful at scale.
   */
  function driftReport(reference, current, features, psiThreshold, ksStatThreshold) {
    features = features || DEPLOY_FEATURES;
    psiThreshold = psiThreshold === undefined ? 0.2 : psiThreshold;
    ksStatThreshold = ksStatThreshold === undefined ? 0.1 : ksStatThreshold;
    return features.map((f) => {
      const ref = dropNaN(reference.map((r) => r[f]));
      const cur = dropNaN(current.map((r) => r[f]));
      const psi = populationStabilityIndex(ref, cur);
      const ks = ksStat(ref, cur);
      return {
        feature: f,
        psi: psi,
        ks_stat: ks,
        ref_mean: mean(ref),
        cur_mean: mean(cur),
        drifted: (psi > psiThreshold) || (ks > ksStatThreshold),
      };
    }).sort((p, q) => (Number.isNaN(q.psi) ? -1 : 0) || q.psi - p.psi);
  }

  // ------------------------------------------------ performance over time --
  function precisionScore(y, pred) {
    let tp = 0, fp = 0;
    for (let i = 0; i < y.length; i++) {
      if (pred[i] === 1) { if (y[i] === 1) tp++; else fp++; }
    }
    return tp + fp === 0 ? 0 : tp / (tp + fp);  // zero_division=0
  }

  function recallScore(y, pred) {
    let tp = 0, fn = 0;
    for (let i = 0; i < y.length; i++) {
      if (y[i] === 1) { if (pred[i] === 1) tp++; else fn++; }
    }
    return tp + fn === 0 ? 0 : tp / (tp + fn);  // zero_division=0
  }

  /** sklearn.metrics.average_precision_score: sum over thresholds of (R_n - R_n-1) * P_n. */
  function averagePrecision(y, s) {
    const n = y.length;
    const nPos = y.reduce((a, v) => a + (v === 1 ? 1 : 0), 0);
    if (nPos === 0 || nPos === n) return NaN;
    const idx = Array.from({ length: n }, (_, i) => i).sort((i, j) => s[j] - s[i]);
    let tp = 0, fp = 0, ap = 0, prevRecall = 0;
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      if (y[i] === 1) tp++; else fp++;
      // only record at a distinct-score boundary (ties share one operating point)
      if (k === n - 1 || s[idx[k + 1]] !== s[i]) {
        const recall = tp / nPos;
        const precision = tp / (tp + fp);
        ap += (recall - prevRecall) * precision;
        prevRecall = recall;
      }
    }
    return ap;
  }

  /** pandas.cut(x, bins=n) edges: equal width over [min, max], first edge nudged down. */
  function cutEdges(values, nBins) {
    let mn = Math.min.apply(null, values), mx = Math.max.apply(null, values);
    if (mn === mx) {
      const adj = mn !== 0 ? Math.abs(mn) * 0.001 : 0.001;
      mn -= adj; mx += adj;
    }
    const edges = [];
    for (let i = 0; i <= nBins; i++) edges.push(mn + ((mx - mn) * i) / nBins);
    if (Math.min.apply(null, values) !== Math.max.apply(null, values)) {
      edges[0] -= (mx - mn) * 0.001;   // pandas widens the first (a, b] interval
    }
    return edges;
  }

  /**
   * Precision / recall / PR-AUC on nWindows equal-width buckets of `timeCol`.
   * Simulates production monitoring where labels arrive over time: each bucket is an
   * operational window scored at the fixed deployed threshold.
   */
  function rollingPerformance(rows, timeCol, yCol, scoreCol, threshold, nWindows) {
    nWindows = nWindows || 12;
    const d = rows.filter((r) =>
      !Number.isNaN(r[timeCol]) && !Number.isNaN(r[yCol]) && !Number.isNaN(r[scoreCol]));
    if (!d.length) return [];
    const edges = cutEdges(d.map((r) => r[timeCol]), nWindows);
    const buckets = Array.from({ length: nWindows }, () => []);
    for (const r of d) {
      // pandas.cut intervals are right-closed: (edge_i, edge_i+1]
      let b = 0;
      while (b < nWindows - 1 && r[timeCol] > edges[b + 1]) b++;
      buckets[b].push(r);
    }
    const out = [];
    for (let b = 0; b < nWindows; b++) {
      const g = buckets[b];
      if (!g.length) continue;   // observed=True drops empty buckets
      const y = g.map((r) => (r[yCol] ? 1 : 0));
      const s = g.map((r) => r[scoreCol]);
      const pred = s.map((v) => (v >= threshold ? 1 : 0));
      const nFraud = y.reduce((a, v) => a + v, 0);
      out.push({
        window_start: edges[b],
        window_end: edges[b + 1],
        n: g.length,
        n_fraud: nFraud,
        fraud_rate: nFraud / g.length,
        precision: precisionScore(y, pred),
        recall: recallScore(y, pred),
        pr_auc: averagePrecision(y, s),
      });
    }
    return out;
  }

  // ------------------------------------------------------ retrain triggers --
  /** Turn drift + rolling-performance evidence into a concrete retrain / hold decision. */
  function evaluateTriggers(driftRows, perfRows, refPrAuc, cfg) {
    cfg = cfg || DEFAULT_TRIGGERS;
    const fired = [];

    if (driftRows.length) {
      const nDrift = driftRows.filter((r) => r.drifted).length;
      const valid = driftRows.filter((r) => !Number.isNaN(r.psi));
      if (valid.length) {
        let worstRow = valid[0];
        for (const r of valid) if (r.psi > worstRow.psi) worstRow = r;
        if (worstRow.psi > cfg.psi_significant) {
          fired.push(`Data drift: max PSI ${worstRow.psi.toFixed(3)} > ` +
            `${cfg.psi_significant} on '${worstRow.feature}'.`);
        }
      }
      if (nDrift >= cfg.n_features_drifted) {
        fired.push(`Breadth of drift: ${nDrift} served features flagged ` +
          `(>= ${cfg.n_features_drifted}).`);
      }
    }

    if (perfRows.length) {
      const minRecall = Math.min.apply(null, perfRows.map((r) => r.recall));
      const minPrec = Math.min.apply(null, perfRows.map((r) => r.precision));
      const prs = perfRows.map((r) => r.pr_auc).filter((v) => !Number.isNaN(v));
      const minPr = prs.length ? Math.min.apply(null, prs) : NaN;
      if (minRecall < cfg.recall_floor) {
        fired.push(`Recall breach: rolling recall fell to ${minRecall.toFixed(3)} ` +
          `< floor ${cfg.recall_floor}.`);
      }
      if (minPrec < cfg.precision_floor) {
        fired.push(`Precision breach: rolling precision fell to ${minPrec.toFixed(3)} ` +
          `< floor ${cfg.precision_floor}.`);
      }
      if (!Number.isNaN(minPr) && refPrAuc - minPr > cfg.pr_auc_drop) {
        fired.push(`PR-AUC decay: rolling PR-AUC fell to ${minPr.toFixed(3)}, ` +
          `${(refPrAuc - minPr).toFixed(3)} below reference ${refPrAuc.toFixed(3)}.`);
      }
    }

    return { retrain_recommended: fired.length > 0, reasons: fired };
  }

  // ---------------------------------------------------- stream simulation --
  /* A SIMULATION, not a production feed — the same honest stand-in that
   * build_monitoring._synthetic_stream() uses when real exports are absent. What is real
   * here is the model doing the scoring and the monitoring maths above.
   *
   * Transactions are PaySim-shaped so the deployed model's scores are meaningful:
   *   legit — balances reconcile, so errorBalanceOrig / errorBalanceDest sit at ~0
   *   fraud — the account is drained (amount == oldbalanceOrg, newbalanceOrig == 0) into a
   *           destination whose balance never moves, leaving errorBalanceDest == amount
   * The "degraded" regime models an upstream ledger fault: larger amounts, and balance
   * bookkeeping that stops reconciling on ordinary traffic — which is precisely the
   * footprint the model learnt to treat as fraud, so precision falls as PSI climbs.
   */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Both regimes carry the SAME fraud rate, so every change the dashboard reports is
   * attributable to the ledger fault rather than to a shift in fraud prevalence.
   * That rate is 1%, deliberately above PaySim's post-filter 0.296%, so the rolling
   * windows collect enough positives to give a readable precision/recall in seconds —
   * the page states this. */
  const REGIMES = {
    steady:   { amountScale: 1.0e5, ledgerFaultRate: 0.00, fraudRate: 0.010 },
    degraded: { amountScale: 2.4e5, ledgerFaultRate: 0.55, fraudRate: 0.010 },
  };

  function makeStream(seed) {
    const rnd = mulberry32(seed === undefined ? 7 : seed);
    const normal = () => {
      const u = Math.max(rnd(), 1e-12), v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    const lognormal = (median, sigma) => median * Math.exp(sigma * normal());
    const uniform = (lo, hi) => lo + (hi - lo) * rnd();

    /** One PaySim-shaped transaction under the named regime. */
    function draw(regimeName, id) {
      const cfg = REGIMES[regimeName] || REGIMES.steady;
      const isFraud = rnd() < cfg.fraudRate ? 1 : 0;
      const type = rnd() < 0.5 ? "TRANSFER" : "CASH_OUT";
      let amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest;

      if (isFraud) {
        // drain the origin account into a destination that never credits it
        oldbalanceOrg = lognormal(cfg.amountScale, 1.1);
        amount = oldbalanceOrg;
        newbalanceOrig = 0;
        oldbalanceDest = 0;
        newbalanceDest = 0;
      } else {
        amount = lognormal(cfg.amountScale, 1.1);
        oldbalanceOrg = amount * uniform(1.05, 4.0);
        newbalanceOrig = Math.max(oldbalanceOrg - amount, 0);
        oldbalanceDest = lognormal(6.0e4, 1.3);
        newbalanceDest = oldbalanceDest + amount;
        if (rnd() < cfg.ledgerFaultRate) {
          // upstream ledger stops reconciling: the destination credit goes missing
          newbalanceDest = oldbalanceDest + amount * uniform(0.0, 0.45);
        }
      }
      return {
        id: id, type: type, amount: amount,
        oldbalanceOrg: oldbalanceOrg, newbalanceOrig: newbalanceOrig,
        oldbalanceDest: oldbalanceDest, newbalanceDest: newbalanceDest,
        isFraud: isFraud,
      };
    }

    return { draw: draw, regimes: REGIMES };
  }

  /** Derive the served feature row exactly as deployment/scoring.featurize() does. */
  function featurize(t) {
    const a = +t.amount, oo = +t.oldbalanceOrg, no = +t.newbalanceOrig;
    const od = +t.oldbalanceDest, nd = +t.newbalanceDest;
    return {
      amount: a, oldbalanceOrg: oo, newbalanceOrig: no,
      oldbalanceDest: od, newbalanceDest: nd,
      errorBalanceOrig: no + a - oo,
      errorBalanceDest: od + a - nd,
      type_TRANSFER: t.type === "TRANSFER" ? 1 : 0,
    };
  }

  return {
    DEPLOY_FEATURES: DEPLOY_FEATURES,
    DEFAULT_TRIGGERS: DEFAULT_TRIGGERS,
    populationStabilityIndex: populationStabilityIndex,
    ksStat: ksStat,
    driftReport: driftReport,
    averagePrecision: averagePrecision,
    rollingPerformance: rollingPerformance,
    evaluateTriggers: evaluateTriggers,
    makeStream: makeStream,
    featurize: featurize,
  };
});
