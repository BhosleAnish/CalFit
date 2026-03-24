import { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";

// ─── Chart.js via CDN (loaded once) ──────────────────────────────────────────
function useChartJS() {
  const [ready, setReady] = useState(!!window.Chart);
  useEffect(() => {
    if (window.Chart) { setReady(true); return; }
    const s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/chart.js";
    s.onload = () => setReady(true);
    document.head.appendChild(s);
  }, []);
  return ready;
}

// ─── Navbar ───────────────────────────────────────────────────────────────────
function Navbar({ onLogout }) {
  return (
    <header style={s.header}>
      <div style={s.logo}>Cal-FIT</div>
      <nav style={s.mainNav}>
        <Link to="/dashboard" style={s.navLink}>Home</Link>
        <Link to="/scan" style={s.navLink}>Scan</Link>
        <Link to="/profile" style={s.navLink}>Profile</Link>
      </nav>
      <button onClick={onLogout} style={s.logoutBtn}>Logout</button>
    </header>
  );
}

// ─── Stat Item ────────────────────────────────────────────────────────────────
function StatItem({ value, label }) {
  return (
    <div style={s.statItem}>
      <span style={s.statValue}>{value}</span>
      <div style={s.statLabel}>{label}</div>
    </div>
  );
}

// ─── Scan Card ────────────────────────────────────────────────────────────────
function ScanCard({ scan }) {
  const statusColor = { high: "#ef4444", low: "#f59e0b", normal: "#84BF04", unknown: "#9CA3AF" };
  const formatTime = (isoStr) => {
    if (!isoStr) return "";
    try { return new Date(isoStr).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
    catch { return ""; }
  };
  return (
    <div style={s.scanCard}>
      <h4 style={s.scanCardTitle}>{scan.product_name}</h4>
      <div style={s.scanTime}>Scanned at {formatTime(scan.scan_time)}</div>
      <div style={s.nutrientList}>
        {Object.entries(scan.nutrients || {}).map(([name, data]) => (
          <div key={name} style={s.nutrientItem}>
            <span>{name.charAt(0).toUpperCase() + name.slice(1)}:</span>
            <span style={{ fontWeight: "bold", color: statusColor[data.status?.toLowerCase()] || "#252601" }}>
              {parseFloat(data.value).toFixed(1)}{data.unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Dual Charts ─────────────────────────────────────────────────────────────
function DualCharts({ chartReady }) {
  const healthRef = useRef(null);
  const scanRef   = useRef(null);
  const healthChartInstance = useRef(null);
  const scanChartInstance   = useRef(null);

  const [activePeriod, setActivePeriod] = useState("weekly");
  const [healthMsg,    setHealthMsg]    = useState("");
  const [scanMsg,      setScanMsg]      = useState("");
  const [healthLoading, setHealthLoading] = useState(false);
  const [scanLoading,   setScanLoading]   = useState(false);

  // Destroy + recreate canvas to avoid "canvas already in use" errors
  const resetCanvas = (ref, instanceRef) => {
    if (instanceRef.current) {
      instanceRef.current.destroy();
      instanceRef.current = null;
    }
  };

  const renderHealthChart = (data) => {
    if (!data || !data.labels?.length) {
      setHealthMsg("Not enough scan data for this period. Keep scanning!");
      return;
    }
    setHealthMsg("");
    resetCanvas(healthRef, healthChartInstance);
    const ctx = healthRef.current.getContext("2d");
    healthChartInstance.current = new window.Chart(ctx, {
      type: "line",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Your Diet Score",
            data: data.actual_scores,
            borderColor: "#10B981",
            backgroundColor: "rgba(16,185,129,0.1)",
            fill: true, tension: 0.3, borderWidth: 2
          },
          {
            label: "Goal Score",
            data: data.goal_scores,
            borderColor: "#F59E0B",
            borderDash: [5, 5],
            fill: false, borderWidth: 2, pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        scales: {
          y: {
            beginAtZero: true, max: 100,
            ticks: { color: "#9CA3AF" },
            grid: { color: "rgba(255,255,255,0.1)" }
          },
          x: {
            ticks: { color: "#9CA3AF", maxRotation: 45 },
            grid: { color: "rgba(255,255,255,0.1)" }
          }
        },
        plugins: {
          legend: { labels: { color: "#E5E7EB" } },
          tooltip: { backgroundColor: "#1F2937", titleColor: "#E5E7EB", bodyColor: "#D1D5DB" }
        }
      }
    });
  };

  const renderScanChart = (data) => {
    if (!data || !data.labels?.length) {
      setScanMsg("No scan activity data available for this period.");
      return;
    }
    setScanMsg("");
    resetCanvas(scanRef, scanChartInstance);
    const ctx = scanRef.current.getContext("2d");
    scanChartInstance.current = new window.Chart(ctx, {
      type: "bar",
      data: {
        labels: data.labels,
        datasets: [{
          label: "Scans",
          data: data.scan_counts,
          backgroundColor: "rgba(59,130,246,0.7)",
          borderColor: "#3B82F6",
          borderWidth: 1, borderRadius: 4, borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { color: "#9CA3AF", stepSize: 1 },
            grid: { color: "rgba(255,255,255,0.1)" }
          },
          x: {
            ticks: { color: "#9CA3AF", maxRotation: 45 },
            grid: { color: "rgba(255,255,255,0.1)" }
          }
        },
        plugins: {
          legend: { labels: { color: "#E5E7EB" } },
          tooltip: {
            backgroundColor: "#1F2937", titleColor: "#E5E7EB", bodyColor: "#D1D5DB",
            callbacks: { label: (ctx) => `Scans: ${ctx.parsed.y}` }
          }
        }
      }
    });
  };

  const updateCharts = async (period) => {
    if (!chartReady) return;
    setActivePeriod(period);
    setHealthLoading(true); setScanLoading(true);
    setHealthMsg("");      setScanMsg("");

    // Destroy existing charts immediately before fetch
    resetCanvas(healthRef, healthChartInstance);
    resetCanvas(scanRef,   scanChartInstance);

    try {
      const [healthRes, scanRes] = await Promise.all([
        fetch(`/api/health-score/${period}`, { credentials: "include" })
          .then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`/api/scan-count/${period}`,   { credentials: "include" })
          .then(r => r.ok ? r.json() : null).catch(() => null)
      ]);
      renderHealthChart(healthRes);
      renderScanChart(scanRes);
    } catch {
      setHealthMsg("Failed to load chart data.");
      setScanMsg("Failed to load chart data.");
    } finally {
      setHealthLoading(false);
      setScanLoading(false);
    }
  };

  useEffect(() => {
    if (chartReady) updateCharts("weekly");
    return () => {
      healthChartInstance.current?.destroy();
      scanChartInstance.current?.destroy();
    };
  }, [chartReady]);

  return (
    <div style={s.chartsContainer}>
      <div style={s.chartControls}>
        {["daily", "weekly", "monthly"].map(p => (
          <button
            key={p}
            style={{ ...s.timeBtn, ...(activePeriod === p ? s.timeBtnActive : {}) }}
            onClick={() => updateCharts(p)}
          >
            {p.charAt(0).toUpperCase() + p.slice(1)}
          </button>
        ))}
      </div>
      <div style={s.dualCharts}>
        {/* Health Score Chart */}
        <div style={s.chartWrapper}>
          <h3 style={s.chartTitle}>Dietary Health Score Trend</h3>
          <div style={s.chartContainer}>
            {healthLoading && (
              <div style={s.chartLoader}>
                <div style={s.spinner} />
                <p style={{ color: "#E5E7EB" }}>Analyzing health trends...</p>
              </div>
            )}
            {healthMsg && !healthLoading && (
              <p style={s.chartMessage}>{healthMsg}</p>
            )}
            <canvas
              ref={healthRef}
              style={{ display: (healthMsg || healthLoading) ? "none" : "block" }}
            />
          </div>
        </div>
        {/* Scan Count Chart */}
        <div style={s.chartWrapper}>
          <h3 style={s.chartTitle}>Scanning Activity</h3>
          <div style={s.chartContainer}>
            {scanLoading && (
              <div style={s.chartLoader}>
                <div style={s.spinner} />
                <p style={{ color: "#E5E7EB" }}>Loading scan activity...</p>
              </div>
            )}
            {scanMsg && !scanLoading && (
              <p style={s.chartMessage}>{scanMsg}</p>
            )}
            <canvas
              ref={scanRef}
              style={{ display: (scanMsg || scanLoading) ? "none" : "block" }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Main Profile Page ────────────────────────────────────────────────────────
export default function Profile() {
  const navigate = useNavigate();
  const chartReady = useChartJS();

  const [data,      setData]      = useState(null);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);
  const [aiAnalysis, setAiAnalysis] = useState("");
  const [aiVisible,  setAiVisible]  = useState(false);
  const [aiLoading,  setAiLoading]  = useState(false);
  const aiRef = useRef(null);

  useEffect(() => {
    fetch("/api/profile", { credentials: "include" })
      .then(r => {
        if (r.status === 401) { navigate("/"); return null; }
        if (r.status === 404) { navigate("/profile-form"); return null; }
        if (r.status === 400) { navigate("/edit-profile"); return null; }
        return r.json();
      })
      .then(d => { if (d) setData(d); })
      .catch(() => setError("Failed to load profile."))
      .finally(() => setLoading(false));
  }, [navigate]);

  const handleLogout = async () => {
    try { await fetch("/api/logout", { method: "POST", credentials: "include" }); } catch (_) {}
    navigate("/");
  };

  const handleAiAnalysis = async () => {
    if (aiVisible) { setAiVisible(false); return; }
    setAiLoading(true);
    try {
      const res  = await fetch("/api/ai-analysis", { method: "POST", credentials: "include" });
      const data = await res.json();
      if (data.success) {
        setAiAnalysis(data.analysis);
        setAiVisible(true);
        setTimeout(() => aiRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
      }
    } catch {
      alert("Failed to get analysis. Please try again.");
    } finally {
      setAiLoading(false);
    }
  };

  const formatDate = (isoStr) => {
    if (!isoStr) return "N/A";
    try { return new Date(isoStr).toLocaleDateString([], { month: "2-digit", day: "2-digit" }); }
    catch { return "N/A"; }
  };

  if (loading) return (
    <div style={{ ...s.body, display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
      <div style={s.spinner} />
    </div>
  );

  if (error) return (
    <div style={s.body}>
      <Navbar onLogout={handleLogout} />
      <div style={{ textAlign: "center", padding: 60, color: "#ef4444" }}>{error}</div>
    </div>
  );

  const { profile, scan_stats, today_scans, intake, recommendation, percentages } = data || {};

  return (
    <div style={s.body}>
      <Navbar onLogout={handleLogout} />
      <div style={s.container}>
        <h2 style={s.pageTitle}>Your Nutritional Overview</h2>

        {/* ── Charts ── */}
        <div style={s.topSection}>
          <DualCharts chartReady={chartReady} />
        </div>

        {/* ── Profile Grid ── */}
        <div style={s.profileGrid}>
          {/* Left — Scan Stats */}
          <div style={s.leftCol}>
            <div style={s.card}>
              <h3 style={s.cardTitle}>Your Scanning Activity</h3>
              {scan_stats?.has_scans ? (
                <div style={s.statsGrid}>
                  <StatItem value={scan_stats.total_scans}  label="Total Scans" />
                  <StatItem value={scan_stats.recent_scans} label="This Week" />
                  <StatItem value={today_scans?.length || 0} label="Today" />
                  {scan_stats.latest_scan_date && (
                    <StatItem value={formatDate(scan_stats.latest_scan_date)} label="Last Scan" />
                  )}
                </div>
              ) : (
                <div style={s.noScans}>
                  <p>No scans recorded yet.</p>
                  <Link to="/scan" style={s.greenLink}>Start scanning!</Link>
                </div>
              )}
            </div>
          </div>

          {/* Right — Today's Scans */}
          <div style={s.rightCol}>
            <div style={s.card}>
              <h3 style={s.cardTitle}>Today's Food Scans</h3>
              {today_scans?.length > 0 ? (
                <div style={s.scanCards}>
                  {today_scans.map((scan, i) => <ScanCard key={i} scan={scan} />)}
                </div>
              ) : (
                <div style={s.noScans}>
                  <p>No scans recorded for today yet.</p>
                  <Link to="/scan"><button style={s.scanBtn}>Start Scanning</button></Link>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── AI Analysis ── */}
        <div style={s.aiSection}>
          <button
            style={{ ...s.aiBtn, opacity: aiLoading ? 0.7 : 1 }}
            onClick={handleAiAnalysis}
            disabled={aiLoading}
          >
            {aiLoading
              ? <><div style={{ ...s.spinner, width: 20, height: 20, marginRight: 8 }} />Analyzing...</>
              : aiVisible ? "Hide Analysis" : "Get Comprehensive AI Health Analysis"
            }
          </button>
          <p style={{ textAlign: "center", fontSize: 14, marginTop: 10, color: (scan_stats?.total_scans || 0) >= 10 ? "#10B981" : "#9CA3AF" }}>
            {(scan_stats?.total_scans || 0) >= 10
              ? `Analysis available (${scan_stats.total_scans}/10) ✓`
              : `Scan at least 10 food items for analysis (${scan_stats?.total_scans || 0}/10)`
            }
          </p>
          {aiVisible && (
            <div ref={aiRef} style={s.aiContent}
              dangerouslySetInnerHTML={{ __html: aiAnalysis }} />
          )}
        </div>

        {/* ── Actions ── */}
        <div style={s.actions}>
          <Link to="/edit-profile" style={s.actionBtn}>Edit Profile</Link>
          <Link to="/my-scans"    style={s.actionBtn}>View All Scans</Link>
          <button onClick={handleLogout} style={{ ...s.actionBtn, border: "none", cursor: "pointer" }}>Logout</button>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        .analysis-section {
          background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(217,214,208,0.95) 100%);
          backdrop-filter: blur(15px);
          border: 2px solid rgba(132,191,4,0.2);
          border-radius: 20px;
          padding: 25px 30px;
          margin-bottom: 25px;
          transition: all 0.3s ease;
          box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .analysis-section:hover { transform: translateY(-5px); border-color: rgba(132,191,4,0.4); box-shadow: 0 10px 30px rgba(132,191,4,0.2); }
        .analysis-section h3 { color: #84BF04; font-size: 1.4em; margin-bottom: 18px; padding-bottom: 12px; border-bottom: 2px solid rgba(132,191,4,0.3); font-weight: 700; }
        .analysis-section p { color: #252601; line-height: 1.8; margin-bottom: 15px; font-size: 0.95em; font-weight: 500; }
        .analysis-section ul, .analysis-section ol { color: #252601; margin-left: 25px; margin-bottom: 15px; line-height: 1.8; }
        .analysis-section li { margin-bottom: 10px; padding-left: 5px; font-weight: 500; }
        .analysis-section ul li::marker { color: #84BF04; }
        .analysis-section strong { color: #84BF04; font-weight: 700; }
      `}</style>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = {
  body: {
    fontFamily: "'Montserrat', sans-serif",
    minHeight: "100vh",
    background: "linear-gradient(135deg, #D9D6D0 0%, #c4c1bb 100%)",
    color: "#252601",
    overflowX: "hidden",
  },
  header: {
    position: "sticky", top: 0, height: 80, zIndex: 1000,
    background: "linear-gradient(135deg, #84BF04 0%, #9dd305 100%)",
    padding: "0 40px", display: "flex", justifyContent: "space-between", alignItems: "center",
    boxShadow: "0 4px 20px rgba(132,191,4,0.3)", borderBottom: "2px solid rgba(37,38,1,0.1)",
  },
  logo: { fontSize: 28, fontWeight: "bold", color: "#252601", letterSpacing: 1 },
  mainNav: {
    position: "absolute", left: "50%", transform: "translateX(-50%)",
    display: "flex", gap: 30, background: "rgba(37,38,1,0.05)",
    padding: "10px 25px", borderRadius: 50, backdropFilter: "blur(10px)",
  },
  navLink: { color: "#252601", textDecoration: "none", fontSize: 16, fontWeight: 600, padding: "10px 18px", borderRadius: 25, transition: "all 0.3s ease" },
  logoutBtn: {
    background: "rgba(37,38,1,0.15)", border: "2px solid rgba(37,38,1,0.3)",
    color: "#252601", padding: "8px 20px", borderRadius: 25, fontWeight: 600,
    cursor: "pointer", fontFamily: "'Montserrat', sans-serif", fontSize: 14,
  },
  container: { maxWidth: 1400, margin: "40px auto", padding: "0 25px", position: "relative", zIndex: 1 },
  pageTitle: { textAlign: "center", marginBottom: 40, color: "#84BF04", fontSize: 48, fontWeight: 700, textShadow: "2px 2px 4px rgba(132,191,4,0.2)" },
  topSection: { marginBottom: 40 },
  chartsContainer: {
    background: "linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(217,214,208,0.95) 100%)",
    backdropFilter: "blur(15px)", borderRadius: 20, padding: 30,
    boxShadow: "0 10px 40px rgba(0,0,0,0.15)", border: "2px solid rgba(132,191,4,0.2)",
  },
  chartControls: { marginBottom: 25, textAlign: "center", paddingBottom: 20, borderBottom: "2px solid rgba(132,191,4,0.2)" },
  timeBtn: {
    background: "rgba(132,191,4,0.1)", border: "2px solid rgba(132,191,4,0.3)",
    color: "#252601", padding: "10px 20px", margin: "0 8px", borderRadius: 25,
    cursor: "pointer", fontWeight: 600, transition: "all 0.3s ease",
    fontFamily: "'Montserrat', sans-serif",
  },
  timeBtnActive: { background: "linear-gradient(135deg,#84BF04,#9dd305)", borderColor: "#84BF04", transform: "translateY(-2px)", boxShadow: "0 4px 15px rgba(132,191,4,0.3)" },
  dualCharts: { display: "flex", gap: 25 },
  chartWrapper: {
    flex: 1, background: "#0D0D0D", borderRadius: 15, padding: 20,
    border: "2px solid rgba(132,191,4,0.3)", position: "relative", overflow: "hidden",
  },
  chartTitle: { color: "#84BF04", marginBottom: 15, fontSize: "1.2em", textAlign: "center", fontWeight: 700 },
  chartContainer: { height: 300, position: "relative", minHeight: 300 },
  chartLoader: { position: "absolute", inset: 0, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", background: "rgba(13,13,13,0.95)", borderRadius: 8, zIndex: 2, gap: 12 },
  chartMessage: { textAlign: "center", marginTop: 15, fontSize: "0.95em", color: "#9CA3AF", padding: 20, fontWeight: 500 },
  profileGrid: { display: "flex", gap: 30, flexWrap: "wrap", marginBottom: 40 },
  leftCol: { flex: 1, minWidth: 300 },
  rightCol: { flex: 1, minWidth: 300 },
  card: {
    background: "linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(217,214,208,0.95) 100%)",
    backdropFilter: "blur(15px)", padding: 25, borderRadius: 20,
    boxShadow: "0 10px 40px rgba(0,0,0,0.15)", border: "2px solid rgba(132,191,4,0.2)",
    height: "100%",
  },
  cardTitle: { color: "#84BF04", marginBottom: 20, fontSize: "1.3em", fontWeight: 700 },
  statsGrid: { display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 15, marginTop: 15 },
  statItem: { textAlign: "center", background: "rgba(255,255,255,0.6)", padding: 20, borderRadius: 12, border: "2px solid rgba(132,191,4,0.15)", transition: "all 0.3s ease" },
  statValue: { display: "block", fontSize: "2em", fontWeight: "bold", color: "#84BF04", marginBottom: 8 },
  statLabel: { fontSize: "0.9em", color: "#666", fontWeight: 600 },
  scanCards: { display: "flex", flexDirection: "column", gap: 15, maxHeight: 400, overflowY: "auto" },
  scanCard: { background: "rgba(255,255,255,0.6)", padding: 18, borderRadius: 12, border: "2px solid rgba(132,191,4,0.15)", transition: "all 0.3s ease" },
  scanCardTitle: { marginBottom: 8, color: "#84BF04", fontWeight: 700, fontSize: "1.1em" },
  scanTime: { fontSize: "0.85em", color: "#666", marginBottom: 10, fontWeight: 500 },
  nutrientList: { display: "flex", flexDirection: "column", gap: 8 },
  nutrientItem: { display: "flex", justifyContent: "space-between", fontSize: "0.9em", color: "#252601" },
  noScans: { textAlign: "center", color: "#666", padding: 30 },
  greenLink: { color: "#84BF04", textDecoration: "none", fontWeight: 600 },
  scanBtn: {
    background: "linear-gradient(135deg,#84BF04,#9dd305)", color: "#252601",
    border: "none", padding: "12px 24px", borderRadius: 25, fontSize: "1rem",
    fontWeight: 700, cursor: "pointer", boxShadow: "0 4px 15px rgba(132,191,4,0.3)",
    marginTop: 15, fontFamily: "'Montserrat', sans-serif",
  },
  aiSection: { margin: "40px 0", textAlign: "center" },
  aiBtn: {
    background: "linear-gradient(135deg,#84BF04,#9dd305)", border: "none",
    color: "#252601", padding: "14px 30px", fontSize: "1.1em", fontWeight: 700,
    borderRadius: 50, cursor: "pointer", display: "inline-flex", alignItems: "center",
    gap: 12, boxShadow: "0 4px 15px rgba(132,191,4,0.3)", transition: "all 0.3s ease",
    fontFamily: "'Montserrat', sans-serif",
  },
  aiContent: { marginTop: 30, textAlign: "left" },
  actions: { display: "flex", justifyContent: "center", gap: 20, marginTop: 40, flexWrap: "wrap" },
  actionBtn: {
    background: "linear-gradient(135deg,#84BF04,#9dd305)", color: "#252601",
    textDecoration: "none", padding: "14px 28px", borderRadius: 50,
    fontWeight: 700, fontSize: 16, boxShadow: "0 4px 15px rgba(132,191,4,0.3)",
    transition: "all 0.3s ease", fontFamily: "'Montserrat', sans-serif",
    display: "inline-block",
  },
  spinner: {
    border: "4px solid rgba(132,191,4,0.2)", borderTop: "4px solid #84BF04",
    borderRadius: "50%", width: 40, height: 40,
    animation: "spin 1s linear infinite",
  },
};