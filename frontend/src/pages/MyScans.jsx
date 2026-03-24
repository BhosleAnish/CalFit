import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600;700&display=swap');

  .ms-root {
    min-height: 100vh;
    background: #f4f1eb;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a0e;
  }

  .ms-header {
    background: linear-gradient(135deg, #84BF04 0%, #9dd305 100%);
    padding: 0 40px;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 20px rgba(132,191,4,0.3);
  }

  .ms-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: #1a1a0e;
    letter-spacing: 3px;
  }

  .ms-nav { display: flex; gap: 8px; }

  .ms-nav a {
    padding: 8px 20px;
    border-radius: 50px;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    color: #1a1a0e;
    transition: all 0.25s ease;
  }

  .ms-nav a:hover,
  .ms-nav a.active {
    background: #1a1a0e;
    color: #84BF04;
  }

  .ms-body {
    max-width: 1400px;
    margin: 0 auto;
    padding: 40px 24px 80px;
  }

  .ms-back {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 36px;
    padding: 10px 24px;
    border-radius: 50px;
    border: 2px solid #84BF04;
    background: white;
    color: #1a1a0e;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.25s ease;
  }

  .ms-back:hover {
    background: #84BF04;
    transform: translateX(-4px);
  }

  .ms-hero {
    background: #1a1a0e;
    border-radius: 24px;
    padding: 56px 48px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
  }

  .ms-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(132,191,4,0.18) 0%, transparent 70%);
    pointer-events: none;
  }

  .ms-hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 72px;
    color: #84BF04;
    letter-spacing: 4px;
    margin: 0 0 8px;
    line-height: 1;
  }

  .ms-hero p {
    color: #a8a89a;
    font-size: 17px;
    margin: 0;
  }

  .ms-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 36px;
  }

  .ms-stat {
    background: white;
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }

  .ms-stat:hover {
    border-color: #84BF04;
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(132,191,4,0.2);
  }

  .ms-stat-num {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 56px;
    color: #84BF04;
    line-height: 1;
    letter-spacing: 2px;
  }

  .ms-stat-label {
    font-size: 13px;
    font-weight: 700;
    color: #6b6b5a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
  }

  .ms-filters {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 32px;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }

  .ms-filters input,
  .ms-filters select {
    flex: 1;
    min-width: 200px;
    padding: 11px 18px;
    border: 2px solid #e8e4dc;
    border-radius: 50px;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    background: #f9f7f3;
    color: #1a1a0e;
    transition: border-color 0.2s;
    outline: none;
  }

  .ms-filters input:focus,
  .ms-filters select:focus {
    border-color: #84BF04;
    background: white;
  }

  .ms-btn {
    padding: 11px 26px;
    border-radius: 50px;
    border: none;
    cursor: pointer;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    transition: all 0.25s ease;
  }

  .ms-btn-primary {
    background: linear-gradient(135deg, #84BF04, #9dd305);
    color: #1a1a0e;
    box-shadow: 0 4px 14px rgba(132,191,4,0.35);
  }

  .ms-btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(132,191,4,0.5);
  }

  .ms-btn-secondary {
    background: #1a1a0e;
    color: #84BF04;
  }

  .ms-btn-secondary:hover {
    background: #333320;
    transform: translateY(-2px);
  }

  .ms-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 24px;
  }

  .ms-card {
    background: white;
    border-radius: 20px;
    overflow: hidden;
    border: 2px solid transparent;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
  }

  .ms-card:hover {
    border-color: #84BF04;
    transform: translateY(-8px);
    box-shadow: 0 16px 40px rgba(132,191,4,0.22);
  }

  .ms-card-head {
    background: linear-gradient(135deg, #84BF04, #9dd305);
    padding: 22px 24px;
    border-bottom: 3px solid #1a1a0e;
  }

  .ms-card-head h3 {
    font-size: 19px;
    font-weight: 700;
    color: #1a1a0e;
    margin: 0 0 6px;
  }

  .ms-card-date {
    font-size: 13px;
    color: rgba(26,26,14,0.7);
    font-weight: 600;
  }

  .ms-card-body { padding: 22px 24px; }

  .ms-nutrients {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 18px;
  }

  .ms-nutrient {
    background: #f4f1eb;
    border: 1px solid rgba(132,191,4,0.25);
    border-radius: 12px;
    padding: 13px;
    text-align: center;
    transition: all 0.2s;
  }

  .ms-nutrient:hover {
    background: rgba(132,191,4,0.1);
    transform: scale(1.04);
  }

  .ms-nutrient strong {
    display: block;
    font-size: 20px;
    font-weight: 700;
    color: #1a1a0e;
    margin-bottom: 3px;
  }

  .ms-nutrient span {
    font-size: 12px;
    color: #6b6b5a;
    font-weight: 600;
    text-transform: capitalize;
  }

  .ms-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 18px;
  }

  .ms-badge {
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 12px;
    font-weight: 700;
  }

  .ms-badge-high { background: linear-gradient(135deg, #ff6b6b, #ff5252); color: white; }
  .ms-badge-normal { background: linear-gradient(135deg, #51cf66, #37b24d); color: white; }
  .ms-badge-low { background: linear-gradient(135deg, #ffd43b, #fab005); color: #1a1a0e; }

  .ms-actions { display: flex; gap: 10px; }

  .ms-act {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    font-weight: 700;
    font-size: 13px;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.25s ease;
  }

  .ms-act-view { background: linear-gradient(135deg,#84BF04,#9dd305); color: #1a1a0e; }
  .ms-act-view:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(132,191,4,0.4); }

  .ms-act-report { background: #ffc107; color: #1a1a0e; }
  .ms-act-report:hover { background: #e0a800; transform: translateY(-2px); }

  .ms-act-delete { background: linear-gradient(135deg,#ff6b6b,#ff5252); color: white; }
  .ms-act-delete:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(255,82,82,0.4); }

  .ms-empty {
    background: white;
    border-radius: 24px;
    padding: 80px 40px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
  }

  .ms-empty h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px;
    color: #1a1a0e;
    letter-spacing: 3px;
    margin-bottom: 12px;
  }

  .ms-empty p { color: #6b6b5a; font-size: 17px; margin-bottom: 28px; }

  .ms-scan-link {
    display: inline-block;
    padding: 14px 36px;
    background: linear-gradient(135deg, #84BF04, #9dd305);
    color: #1a1a0e;
    font-weight: 700;
    font-size: 16px;
    text-decoration: none;
    border-radius: 50px;
    box-shadow: 0 6px 22px rgba(132,191,4,0.4);
    transition: all 0.3s ease;
    cursor: pointer;
    border: none;
  }

  .ms-scan-link:hover {
    transform: translateY(-3px) scale(1.04);
    box-shadow: 0 10px 32px rgba(132,191,4,0.55);
  }

  .ms-loading {
    text-align: center;
    padding: 80px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 3px;
    color: #6b6b5a;
  }

  /* Modal */
  .ms-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 1000;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    align-items: center;
    justify-content: center;
  }

  .ms-overlay.open { display: flex; }

  .ms-modal {
    background: white;
    border-radius: 20px;
    padding: 36px;
    width: 90%;
    max-width: 480px;
    position: relative;
    box-shadow: 0 24px 60px rgba(0,0,0,0.3);
    animation: modalIn 0.28s cubic-bezier(0.4,0,0.2,1);
  }

  @keyframes modalIn {
    from { opacity: 0; transform: scale(0.93) translateY(12px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }

  .ms-modal h2 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 2px;
    margin: 0 0 24px;
    color: #1a1a0e;
  }

  .ms-close {
    position: absolute;
    top: 16px; right: 20px;
    font-size: 26px;
    cursor: pointer;
    color: #aaa;
    background: none;
    border: none;
    line-height: 1;
    transition: color 0.2s;
  }

  .ms-close:hover { color: #1a1a0e; }

  .ms-form-group { margin-bottom: 20px; }

  .ms-form-group label {
    display: block;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 7px;
    color: #333;
  }

  .ms-form-group textarea {
    width: 100%;
    padding: 11px 14px;
    border: 2px solid #e8e4dc;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    resize: vertical;
    box-sizing: border-box;
    outline: none;
    transition: border-color 0.2s;
  }

  .ms-form-group textarea:focus { border-color: #84BF04; }

  .ms-radio-group { display: flex; gap: 16px; flex-wrap: wrap; }

  .ms-radio-group label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
    cursor: pointer;
  }

  .ms-submit {
    width: 100%;
    padding: 13px;
    background: linear-gradient(135deg, #84BF04, #9dd305);
    color: #1a1a0e;
    border: none;
    border-radius: 50px;
    font-weight: 700;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 4px 14px rgba(132,191,4,0.35);
  }

  .ms-submit:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(132,191,4,0.5);
  }

  .ms-status { margin-top: 14px; font-weight: 700; font-size: 14px; }
  .ms-status.success { color: #37b24d; }
  .ms-status.error { color: #ff5252; }

  @media (max-width: 768px) {
    .ms-hero h1 { font-size: 48px; }
    .ms-stats { grid-template-columns: 1fr; }
    .ms-grid { grid-template-columns: 1fr; }
    .ms-filters { flex-direction: column; }
    .ms-header { padding: 0 16px; }
    .ms-body { padding: 24px 16px 60px; }
  }
`;

export default function MyScans() {
  const navigate = useNavigate();
  const [allScans, setAllScans] = useState([]);
  const [displayed, setDisplayed] = useState([]);
  const [stats, setStats] = useState({ total_scans: 0, recent_scans: 0, high_risk_count: 0 });
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState("date-desc");

  // Report modal state
  const [modal, setModal] = useState({ open: false, scanId: null });
  const [reportDesc, setReportDesc] = useState("");
  const [severity, setSeverity] = useState("");
  const [contactConsent, setContactConsent] = useState(false);
  const [reportStatus, setReportStatus] = useState({ msg: "", cls: "" });

  const getCsrf = () =>
    document.querySelector('meta[name="csrf-token"]')?.getAttribute("content") || "";

  const loadScans = useCallback(async () => {
    try {
      const res = await fetch("/api/get-all-scans");
      const data = await res.json();
      if (data.error) { navigate("/"); return; }
      setAllScans(data.scans);
      setDisplayed(data.scans);
      setStats(data.stats || {});
    } catch {
      setDisplayed(null);
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  useEffect(() => { loadScans(); }, [loadScans]);

  const applyFilters = useCallback((s = search, so = sort, scans = allScans) => {
    let filtered = [...scans];
    if (s) filtered = filtered.filter(sc =>
      (sc.product_info?.name || "").toLowerCase().includes(s.toLowerCase())
    );
    filtered.sort((a, b) => {
      switch (so) {
        case "date-desc": return new Date(b.scan_date) - new Date(a.scan_date);
        case "date-asc":  return new Date(a.scan_date) - new Date(b.scan_date);
        case "name-asc":  return (a.product_info?.name || "").localeCompare(b.product_info?.name || "");
        case "name-desc": return (b.product_info?.name || "").localeCompare(a.product_info?.name || "");
        default: return 0;
      }
    });
    setDisplayed(filtered);
  }, [search, sort, allScans]);

  useEffect(() => { applyFilters(search, sort, allScans); }, [search, sort, allScans]);

  const clearFilters = () => { setSearch(""); setSort("date-desc"); };

  const deleteScan = async (scanId) => {
    if (!confirm("Delete this scan? This cannot be undone.")) return;
    try {
      const res = await fetch(`/api/delete-scan/${scanId}`, {
        method: "DELETE",
        credentials: "include",
        headers: { "X-Requested-With": "XMLHttpRequest", "X-CSRFToken": getCsrf() }
      });
      const data = await res.json();
      if (data.success) { alert("✅ Scan deleted!"); loadScans(); }
      else if (data.error === "Unauthorized") { alert("Session expired. Please log in."); navigate("/"); }
      else alert(`⚠️ ${data.error || "Failed to delete"}`);
    } catch { alert("❌ Could not delete scan."); }
  };

  const openModal = (scanId) => {
    setModal({ open: true, scanId });
    setReportDesc(""); setSeverity(""); setContactConsent(false);
    setReportStatus({ msg: "", cls: "" });
  };

  const closeModal = () => setModal({ open: false, scanId: null });

  const submitReport = async () => {
    if (!reportDesc.trim()) { setReportStatus({ msg: "Please describe the issue.", cls: "error" }); return; }
    setReportStatus({ msg: "Submitting...", cls: "" });
    try {
      const res = await fetch(`/api/submit-report/${modal.scanId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": getCsrf(), "X-Requested-With": "XMLHttpRequest" },
        body: JSON.stringify({ description: reportDesc, severity: severity || null, contact_consent: contactConsent })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setReportStatus({ msg: "Report submitted successfully!", cls: "success" });
        setTimeout(closeModal, 2000);
      } else {
        setReportStatus({ msg: `Error: ${data.error || "Could not submit."}`, cls: "error" });
      }
    } catch {
      setReportStatus({ msg: "Network error. Try again.", cls: "error" });
    }
  };

  const formatDate = (d) => new Date(d).toLocaleString();

  const renderCard = (scan) => {
    const name = scan.product_info?.name || "Unknown Product";
    const nutrients = (scan.nutrition_analysis?.structured_nutrients || [])
      .filter(n => ["protein","carbohydrates","carbs","fat","fats","sugar","sodium"].includes(n.nutrient?.toLowerCase()))
      .slice(0, 4);
    const { high_risk_count: hc = 0, normal_count: nc = 0, low_risk_count: lc = 0 } = scan.nutrition_analysis?.summary || {};

    return (
      <div className="ms-card" key={scan._id}>
        <div className="ms-card-head">
          <h3>{name}</h3>
          <div className="ms-card-date">📅 {formatDate(scan.scan_date)}</div>
        </div>
        <div className="ms-card-body">
          {nutrients.length > 0 && (
            <div className="ms-nutrients">
              {nutrients.map((n, i) => (
                <div className="ms-nutrient" key={i}>
                  <strong>{n.value}{n.unit}</strong>
                  <span>{n.nutrient}</span>
                </div>
              ))}
            </div>
          )}
          <div className="ms-badges">
            {hc > 0 && <span className="ms-badge ms-badge-high">⚠️ {hc} High</span>}
            {nc > 0 && <span className="ms-badge ms-badge-normal">✓ {nc} Normal</span>}
            {lc > 0 && <span className="ms-badge ms-badge-low">⚡ {lc} Low</span>}
          </div>
          <div className="ms-actions">
            <button className="ms-act ms-act-view" onClick={() => navigate(`/view-scan/${scan._id}`)}>View Details</button>
            <button className="ms-act ms-act-report" onClick={() => openModal(scan._id)}>Report</button>
            <button className="ms-act ms-act-delete" onClick={() => deleteScan(scan._id)}>Delete</button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <>
      <style>{styles}</style>
      <div className="ms-root">
        {/* Header */}
        <header className="ms-header">
          <div className="ms-logo">Cal-FIT</div>
          <nav className="ms-nav">
            <a href="/dashboard">Home</a>
            <a href="/scan" className="active">Scan</a>
            <a href="/profile">Profile</a>
          </nav>
        </header>

        <div className="ms-body">
          <button className="ms-back" onClick={() => navigate("/profile")}>← Back to Profile</button>

          {/* Hero */}
          <div className="ms-hero">
            <h1>My Nutrition Scans</h1>
            <p>View and manage all your scanned food items</p>
          </div>

          {/* Stats */}
          <div className="ms-stats">
            <div className="ms-stat">
              <div className="ms-stat-num">{stats.total_scans || 0}</div>
              <div className="ms-stat-label">Total Scans</div>
            </div>
            <div className="ms-stat">
              <div className="ms-stat-num">{stats.recent_scans || 0}</div>
              <div className="ms-stat-label">This Week</div>
            </div>
            <div className="ms-stat">
              <div className="ms-stat-num">{stats.high_risk_count || 0}</div>
              <div className="ms-stat-label">High Risk Items</div>
            </div>
          </div>

          {/* Filters */}
          <div className="ms-filters">
            <input
              type="text"
              placeholder="🔍 Search by product name..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
            <select value={sort} onChange={e => setSort(e.target.value)}>
              <option value="date-desc">Newest First</option>
              <option value="date-asc">Oldest First</option>
              <option value="name-asc">Name A–Z</option>
              <option value="name-desc">Name Z–A</option>
            </select>
            <button className="ms-btn ms-btn-secondary" onClick={clearFilters}>Clear</button>
          </div>

          {/* Content */}
          {loading ? (
            <div className="ms-loading">Loading your scans…</div>
          ) : displayed === null ? (
            <div className="ms-empty">
              <h2>Error Loading Scans</h2>
              <p>Could not load your scans. Please try again.</p>
              <button className="ms-scan-link" onClick={loadScans}>Retry</button>
            </div>
          ) : displayed.length === 0 ? (
            <div className="ms-empty">
              <h2>No Scans Yet</h2>
              <p>Start scanning food labels to track your nutrition!</p>
              <button className="ms-scan-link" onClick={() => navigate("/scan")}>Scan Your First Item</button>
            </div>
          ) : (
            <div className="ms-grid">{displayed.map(renderCard)}</div>
          )}
        </div>

        {/* Report Modal */}
        <div className={`ms-overlay${modal.open ? " open" : ""}`} onClick={e => e.target === e.currentTarget && closeModal()}>
          <div className="ms-modal">
            <button className="ms-close" onClick={closeModal}>&times;</button>
            <h2>Report Issue</h2>

            <div className="ms-form-group">
              <label>Describe the issue:</label>
              <textarea
                rows={4}
                placeholder="e.g., Felt nauseous after eating, allergic reaction..."
                value={reportDesc}
                onChange={e => setReportDesc(e.target.value)}
              />
            </div>

            <div className="ms-form-group">
              <label>Severity (Optional):</label>
              <div className="ms-radio-group">
                {["mild","moderate","severe"].map(s => (
                  <label key={s}>
                    <input type="radio" name="severity" value={s} checked={severity === s} onChange={() => setSeverity(s)} />
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </label>
                ))}
              </div>
            </div>

            <div className="ms-form-group">
              <label>
                <input type="checkbox" checked={contactConsent} onChange={e => setContactConsent(e.target.checked)} />
                {" "}Okay to contact you for follow-up? (Optional)
              </label>
            </div>

            <button className="ms-submit" onClick={submitReport}>Submit Report</button>
            {reportStatus.msg && (
              <div className={`ms-status ${reportStatus.cls}`}>{reportStatus.msg}</div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}