import { useState } from "react";
import { Link } from "react-router-dom";
import "./ScanResult.css";

// ─── Navbar ───────────────────────────────────────────────────────────────────
function Navbar() {
  return (
    <header className="sr-header">
      <div className="sr-logo">Cal-FIT</div>
      <nav className="sr-main-nav">
        <Link to="/dashboard">Home</Link>
        <Link to="/scan">Scan</Link>
        <Link to="/profile">Profile</Link>
      </nav>
    </header>
  );
}

// ─── Detail Panel ─────────────────────────────────────────────────────────────
function DetailPanel({ nutrient, onClose }) {
  if (!nutrient) return null;
  return (
    <div className="detail-panel">
      <h3 id="detail-title">Nutrient Info</h3>
      <p><strong>Nutrient:</strong> {nutrient.nutrient}</p>
      <p><strong>Value:</strong> {nutrient.value}</p>
      <p><strong>Unit:</strong> {nutrient.unit}</p>
      <p><strong>Status:</strong> {nutrient.status}</p>
      {nutrient.message && <p>{nutrient.message}</p>}
      <button onClick={onClose}>Close</button>
    </div>
  );
}

// ─── ScanResult ───────────────────────────────────────────────────────────────
export default function ScanResult({ nutrients = [] }) {
  const [selected, setSelected] = useState(null);

  const detailActive = selected !== null;

  return (
    <div className="sr-body">
      <Navbar />
      <div className="sr-container">
        <h2>Nutritional Breakdown</h2>

        {nutrients.length === 0 ? (
          <p style={{ textAlign: "center" }}>No nutrients extracted from the label.</p>
        ) : (
          <div className={`main-content${detailActive ? " detail-active" : ""}`}>
            <div className="scroll-table-wrapper">
              <table className={`table${detailActive ? " slide-left" : ""}`}>
                <thead>
                  <tr>
                    <th>Nutrient</th>
                    <th>Value</th>
                    <th>Unit</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {nutrients.map((n, i) => (
                    <tr
                      key={i}
                      onClick={() => setSelected(n)}
                      className={selected === n ? "row-selected" : ""}
                    >
                      <td>{n.nutrient?.charAt(0).toUpperCase() + n.nutrient?.slice(1)}</td>
                      <td>{n.value}</td>
                      <td>{n.unit}</td>
                      <td>{n.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {detailActive && (
              <DetailPanel nutrient={selected} onClose={() => setSelected(null)} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}