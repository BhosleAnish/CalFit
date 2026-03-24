import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import "../styles/ViewScan.css";

// ─── Status Badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  return (
    <span className={`status-badge status-${status?.toLowerCase() || "unknown"}`}>
      {status}
    </span>
  );
}

// ─── ViewScan ─────────────────────────────────────────────────────────────────
export default function ViewScan() {
  const { scanId } = useParams();
  const navigate   = useNavigate();

  const [scan,    setScan]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  useEffect(() => {
    const fetchScan = async () => {
      try {
        const res = await fetch(`/api/scans/${scanId}`, {
          credentials: "include"
        });
        if (res.status === 401) { navigate("/"); return; }
        const data = await res.json();
        if (!data.success) {
          setError(data.error || "Scan not found.");
        } else {
          setScan(data.scan);
        }
      } catch (err) {
        setError("Could not load scan details.");
      } finally {
        setLoading(false);
      }
    };
    fetchScan();
  }, [scanId, navigate]);

  const handleDelete = async () => {
    if (!window.confirm("⚠️ Are you sure you want to delete this scan? This action cannot be undone.")) return;
    try {
      const res = await fetch(`/api/scans/${scanId}`, {
        method: "DELETE",
        credentials: "include"
      });
      const data = await res.json();
      if (data.success) {
        alert("✅ Scan deleted successfully");
        navigate("/my-scans");
      } else {
        alert("❌ " + (data.error || "Failed to delete scan"));
      }
    } catch {
      alert("❌ An error occurred while deleting the scan");
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return "Unknown date";
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric", month: "long", day: "numeric",
      hour: "numeric", minute: "2-digit", hour12: true,
    });
  };

  if (loading) return (
    <div className="vs-page">
      <div className="vs-container">
        <div className="vs-loading">Loading scan details...</div>
      </div>
    </div>
  );

  if (error) return (
    <div className="vs-page">
      <div className="vs-container">
        <div className="vs-error">{error}</div>
        <Link to="/my-scans" className="back-btn">← Back to My Scans</Link>
      </div>
    </div>
  );

  const nutrients   = scan?.nutrition_analysis?.structured_nutrients || [];
  const summary     = scan?.nutrition_analysis?.summary || {};
  const productInfo = scan?.product_info || {};
  const scanMeta    = scan?.scan_metadata || {};

  const imageUrl = productInfo.image_url
    || (scanMeta.image_filename ? `http://localhost:5000/static/uploads/${scanMeta.image_filename}` : null)
    || "http://localhost:5000/static/images/no-image.png";

  return (
    <div className="vs-page">
      <div className="vs-container">

        {/* Header */}
        <div className="vs-header">
          <h1>📊 Scan Details</h1>
          <Link to="/my-scans" className="back-btn">← Back to My Scans</Link>
        </div>

        {/* Main Grid */}
        <div className="scan-grid">

          {/* Product Card */}
          <div className="card product-card">
            <img src={imageUrl} alt={productInfo.name} className="product-image" />
            <div className="product-name">{productInfo.name}</div>
            <div className="scan-date">📅 Scanned on {formatDate(scan?.scan_date)}</div>

            <div className="scan-meta">
              <div className="meta-item">
                <div className="meta-value">{nutrients.length}</div>
                <div className="meta-label">Nutrients</div>
              </div>
              <div className="meta-item">
                <div className="meta-value">{summary.high_risk_count || 0}</div>
                <div className="meta-label">High Risk</div>
              </div>
            </div>

            <div className="action-buttons">
              <button className="btn btn-delete" onClick={handleDelete}>🗑️ Delete Scan</button>
            </div>
          </div>

          {/* Info Card */}
          <div className="card info-card">
            <h2>Nutrition Summary</h2>

            <div className="summary-stats">
              <div className="stat-box high">
                <div className="stat-number">{summary.high_risk_count || 0}</div>
                <div className="stat-label">High Risk</div>
              </div>
              <div className="stat-box normal">
                <div className="stat-number">{summary.normal_count || 0}</div>
                <div className="stat-label">Normal</div>
              </div>
              <div className="stat-box low">
                <div className="stat-number">{summary.low_risk_count || 0}</div>
                <div className="stat-label">Low</div>
              </div>
              <div className="stat-box unknown">
                <div className="stat-number">{summary.unknown_count || 0}</div>
                <div className="stat-label">Unknown</div>
              </div>
            </div>

            <h2>Detailed Nutrients</h2>
            <table className="nutrients-table">
              <thead>
                <tr>
                  <th>Nutrient</th>
                  <th>Amount</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {nutrients.map((n, i) => (
                  <>
                    <tr key={`row-${i}`}>
                      <td>
                        <strong>{n.nutrient}</strong>
                        {n.category && <span className="nutrient-category">{n.category}</span>}
                      </td>
                      <td><strong>{n.value} {n.unit}</strong></td>
                      <td><StatusBadge status={n.status} /></td>
                    </tr>
                    {n.message && (
                      <tr key={`msg-${i}`}>
                        <td colSpan="3">
                          <div className="nutrient-message">
                            <strong>💡 Health Insight:</strong> {n.message}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}