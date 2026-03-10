import { useState, useRef, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import api from "../api/axios";
import "../styles/Scan.css";

// ─── Load Cropper.js from CDN ─────────────────────────────────────────────────
function useCropperJS() {
  const [ready, setReady] = useState(!!window.Cropper);
  useEffect(() => {
    if (window.Cropper) { setReady(true); return; }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css";
    document.head.appendChild(link);
    const script = document.createElement("script");
    script.src = "https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js";
    script.onload = () => setReady(true);
    document.head.appendChild(script);
  }, []);
  return ready;
}

// ─── Navbar ───────────────────────────────────────────────────────────────────
function Navbar({ onLogout }) {
  return (
    <header className="scan-header">
      <div className="scan-logo">Cal-FIT</div>
      <nav className="scan-main-nav">
        <Link to="/dashboard">Home</Link>
        <Link to="/scan" className="active">Scan</Link>
        <Link to="/profile">Profile</Link>
      </nav>
      <button onClick={onLogout} className="scan-logout-btn">Logout</button>
    </header>
  );
}

// ─── Loading Overlay ──────────────────────────────────────────────────────────
function LoadingOverlay({ visible, text }) {
  return (
    <div className={`loading-overlay${visible ? " active" : ""}`}>
      <div className="loading-content">
        <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="scanGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%"   style={{ stopColor: "#84BF04", stopOpacity: 0 }} />
              <stop offset="50%"  style={{ stopColor: "#84BF04", stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: "#84BF04", stopOpacity: 0 }} />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          <circle cx="100" cy="100" r="90" fill="none" stroke="rgba(132,191,4,0.1)" strokeWidth="2">
            <animate attributeName="r" values="85;95;85" dur="2s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite" />
          </circle>
          <g transform="translate(60,40)">
            <rect x="0" y="0" width="80" height="100" rx="4" fill="rgba(255,255,255,0.1)" stroke="#84BF04" strokeWidth="2" filter="url(#glow)">
              <animate attributeName="opacity" values="0.8;1;0.8" dur="1.5s" repeatCount="indefinite" />
            </rect>
            <path d="M 60 0 L 80 0 L 80 20 Z" fill="rgba(132,191,4,0.3)" stroke="#84BF04" strokeWidth="1" />
            <path d="M 60 0 L 60 20 L 80 20" fill="none" stroke="#84BF04" strokeWidth="1" strokeDasharray="2,2" />
            {[
              { y: 25, w: 50, delay: "0s" },
              { y: 35, w: 60, delay: "0.2s" },
              { y: 45, w: 45, delay: "0.4s" },
              { y: 55, w: 55, delay: "0.6s" },
              { y: 65, w: 40, delay: "0.8s" },
              { y: 75, w: 50, delay: "1s" },
            ].map((bar, i) => (
              <rect key={i} x="10" y={bar.y} width={bar.w} height="3" rx="1.5" fill="rgba(132,191,4,0.4)">
                <animate attributeName="opacity" values="0.4;0.8;0.4" dur="1s" begin={bar.delay} repeatCount="indefinite" />
              </rect>
            ))}
          </g>
          <rect x="60" y="40" width="80" height="4" fill="url(#scanGradient)" opacity="0.8" filter="url(#glow)">
            <animate attributeName="y" values="40;140;40" dur="2s" repeatCount="indefinite" />
          </rect>
          {[
            { cx: 70,  cy: 60,  dur: "2.5s", color: "#84BF04" },
            { cx: 90,  cy: 70,  dur: "2.8s", color: "#9dd305" },
            { cx: 110, cy: 65,  dur: "2.3s", color: "#84BF04" },
            { cx: 130, cy: 75,  dur: "2.6s", color: "#9dd305" },
          ].map((dot, i) => (
            <circle key={i} cx={dot.cx} cy={dot.cy} r="2" fill={dot.color} opacity="0.6">
              <animate attributeName="cy" values={`${dot.cy};${dot.cy + 70};${dot.cy}`} dur={dot.dur} repeatCount="indefinite" />
              <animate attributeName="opacity" values="0;1;0" dur={dot.dur} repeatCount="indefinite" />
            </circle>
          ))}
          <circle cx="100" cy="100" r="95" fill="none" stroke="rgba(132,191,4,0.2)" strokeWidth="1" strokeDasharray="10 5">
            <animateTransform attributeName="transform" type="rotate" from="0 100 100" to="360 100 100" dur="8s" repeatCount="indefinite" />
          </circle>
          <g transform="translate(100,100)" opacity="0.6">
            <circle r="8" fill="none" stroke="#84BF04" strokeWidth="2">
              <animate attributeName="r" values="6;10;6" dur="1s" repeatCount="indefinite" />
            </circle>
            <circle r="3" fill="#84BF04">
              <animate attributeName="opacity" values="1;0.5;1" dur="1s" repeatCount="indefinite" />
            </circle>
          </g>
        </svg>
        <div className="loading-text">{text || "Analyzing your food label..."}</div>
        <div className="loading-subtext">This may take a few moments</div>
        <div className="progress-container">
          <div className="progress-bar" />
        </div>
      </div>
    </div>
  );
}

// ─── Cropper Modal ────────────────────────────────────────────────────────────
function CropperModal({ visible, imageSrc, cropperReady, onApply, onCancel }) {
  const imgRef    = useRef(null);
  const cropperRef = useRef(null);

  useEffect(() => {
    if (!visible || !imageSrc || !cropperReady || !imgRef.current) return;
    cropperRef.current?.destroy();
    cropperRef.current = new window.Cropper(imgRef.current, {
      aspectRatio: NaN, viewMode: 1, background: false,
      autoCropArea: 0.9, responsive: true, restore: true,
      guides: true, center: true, highlight: true,
      cropBoxMovable: true, cropBoxResizable: true,
      toggleDragModeOnDblclick: false,
    });
    return () => { cropperRef.current?.destroy(); cropperRef.current = null; };
  }, [visible, imageSrc, cropperReady]);

  const handleApply = () => {
    if (!cropperRef.current) return;
    cropperRef.current.getCroppedCanvas({
      maxWidth: 2048, maxHeight: 2048,
      fillColor: "#fff", imageSmoothingEnabled: true, imageSmoothingQuality: "high",
    }).toBlob((blob) => {
      onApply(new File([blob], "cropped_image.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.92);
  };

  return (
    <div className={`cropper-modal${visible ? " active" : ""}`}>
      <div className="cropper-container-wrapper">
        <div className="cropper-header">
          <h3>Edit Your Image</h3>
          <button className="close-cropper" onClick={onCancel}>&times;</button>
        </div>
        <div className="cropper-image-container">
          <img ref={imgRef} src={imageSrc} alt="crop" style={{ maxWidth: "100%", display: "block" }} />
        </div>
        <div className="cropper-controls">
          <button className="cropper-btn secondary" onClick={() => cropperRef.current?.rotate(-90)}>↶ Rotate Left</button>
          <button className="cropper-btn secondary" onClick={() => cropperRef.current?.rotate(90)}>↷ Rotate Right</button>
          <button className="cropper-btn secondary" onClick={() => cropperRef.current?.reset()}>↺ Reset</button>
          <button className="cropper-btn" onClick={handleApply}>✓ Done - Proceed to Scan</button>
          <button className="cropper-btn danger" onClick={onCancel}>✕ Cancel</button>
        </div>
      </div>
    </div>
  );
}

// ─── Nutrient Card ────────────────────────────────────────────────────────────
function NutrientCard({ nutrient, expanded, onToggle }) {
  const statusClass = `nutrient-status status-${nutrient.status?.toLowerCase() || "unknown"}`;
  return (
    <div className={`nutrient-card${expanded ? " expanded" : ""}`} onClick={onToggle}>
      <div className="nutrient-name">{nutrient.nutrient}</div>
      <div className="nutrient-value">{nutrient.value} {nutrient.unit}</div>
      <span className={statusClass}>{nutrient.status}</span>
      {expanded && (
        <div className="ai-analysis-section">
          <div className="ai-divider" />
          {nutrient.ai_analysis && (
            <div className="ai-analysis-content">
              <p className="ai-label">Personalized for You:</p>
              <p className="ai-text">{nutrient.ai_analysis}</p>
            </div>
          )}
          {nutrient.message && (
            <div className="general-message">
              <p className="general-label">General Info:</p>
              <p className="general-text">{nutrient.message}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Community Warnings ───────────────────────────────────────────────────────
function CommunityWarnings({ cw }) {
  if (!cw || cw.total_reports === 0) return (
    <div className="no-warnings">
      <span className="check-icon" />
      <span>No community reports available for this product</span>
    </div>
  );
  const level = cw.warning_level || "none";
  return (
    <div className={`community-warnings warning-level-${level}`}>
      <div className="warning-header">
        <span className="warning-icon" />
        <span className="warning-title">Community Insights</span>
        <span className="report-count">(Based on {cw.total_reports} report{cw.total_reports !== 1 ? "s" : ""})</span>
      </div>
      {cw.summary_message && <div className="warning-summary">{cw.summary_message}</div>}
      {cw.insights?.slice(0, 3).map((ins, i) => (
        <div key={i} className={`insight-item severity-${ins.severity}`}>
          <span className="insight-icon" />
          <div className="insight-content">
            <p className="insight-text">{ins.text}</p>
            <span className="insight-category">{ins.category}</span>
          </div>
        </div>
      ))}
      {cw.positive_reports > 0 && (
        <div className="positive-note">
          <span className="positive-icon" />
          <span>{cw.positive_reports} user{cw.positive_reports !== 1 ? "s" : ""} reported positive experiences</span>
        </div>
      )}
    </div>
  );
}

// ─── Results View ─────────────────────────────────────────────────────────────
function ScanResults({ result, onReset }) {
  const [expandedIdx, setExpandedIdx] = useState(null);
  const [addedToDiet, setAddedToDiet] = useState(false);
  const [flash, setFlash] = useState(null);

  const nutrients = result.scan_type === "label"
    ? result.structured_nutrients
    : result.scan_data?.structured_nutrients || [];
  const scanData = result.scan_data || {};

  const handleAddToDiet = async () => {
    try {
      const res = await api.post("/add-to-diet", {
        product_name: scanData.product_name || "Scanned Item",
        product_image_url: scanData.product_image_url || null,
        structured_nutrients: nutrients,
      });
      setFlash({ type: "success", msg: res.data.message });
      setAddedToDiet(true);
      setTimeout(() => setFlash(null), 3000);
    } catch {
      setFlash({ type: "error", msg: "Failed to add to diet." });
      setTimeout(() => setFlash(null), 3000);
    }
  };

  return (
    <div className="results">
      {flash && (
        <div id="flash-messages">
          <div className={`flash ${flash.type}`}><p>{flash.msg}</p></div>
        </div>
      )}

      <h2>{result.scan_type === "barcode" ? "Barcode Scan Results" : "Label Scan Results"}</h2>

      {result.scan_type === "barcode" && (
        <div className="description-section">
          <h3 className="section-title">Description</h3>
          <div className="description-content">
            <div className="product-info-desc">
              {scanData.product_image_url && (
                <img src={scanData.product_image_url} alt={scanData.product_name} className="product-image-desc" />
              )}
              <div className="product-details">
                <div className="product-name-desc">{scanData.product_name}</div>
                <CommunityWarnings cw={result.community_warnings} />
              </div>
            </div>
          </div>
        </div>
      )}

      <h3 className="section-title" style={{ marginTop: 40 }}>Nutritional Information</h3>
      <div className="nutrient-grid">
        {nutrients.map((n, i) => (
          <NutrientCard key={i} nutrient={n} expanded={expandedIdx === i}
            onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)} />
        ))}
      </div>

      <div className="actions">
        {!addedToDiet && (
          <button className="btn" onClick={handleAddToDiet}>Add to Diet</button>
        )}
        <button className="btn" onClick={onReset}>Scan Another Item</button>
        <Link to="/dashboard" className="btn">Back to Dashboard</Link>
      </div>
    </div>
  );
}

// ─── Main Scan Page ───────────────────────────────────────────────────────────
export default function Scan() {
  const navigate     = useNavigate();
  const cropperReady = useCropperJS();

  const [result,       setResult]       = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [loadingText,  setLoadingText]  = useState("");
  const [error,        setError]        = useState("");
  const [cropVisible,  setCropVisible]  = useState(false);
  const [cropSrc,      setCropSrc]      = useState("");

  const barcodeInputRef = useRef();
  const pendingTypeRef  = useRef(null);

  const handleLogout = async () => {
    try { await api.post("/logout"); } catch (_) {}
    navigate("/");
  };

  const openCropper = (type, file) => {
    if (!file) return;
    pendingTypeRef.current = type;
    const reader = new FileReader();
    reader.onload = (e) => { setCropSrc(e.target.result); setCropVisible(true); };
    reader.readAsDataURL(file);
  };

  const handleCropApply = async (croppedFile) => {
    setCropVisible(false);
    const type = pendingTypeRef.current;
    setLoadingText(type === "barcode" ? "Scanning barcode..." : "Analyzing your food label...");
    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append(type === "label" ? "label_image" : "barcode_image", croppedFile);

    try {
      const res = await api.post("/scan-label", formData, { headers: { "Content-Type": "multipart/form-data" } });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || "Scan failed. Please try again.");
    } finally {
      setLoading(false);
      if (barcodeInputRef.current) barcodeInputRef.current.value = "";
    }
  };

  const handleCropCancel = () => {
    setCropVisible(false);
    setCropSrc("");
    if (barcodeInputRef.current) barcodeInputRef.current.value = "";
  };

  // ── Results page ──
  if (result) return (
    <div className="scan-page">
      <Navbar onLogout={handleLogout} />
      <ScanResults result={result} onReset={() => setResult(null)} />
    </div>
  );

  // ── Upload page ──
  return (
    <div className="scan-page">
      <Navbar onLogout={handleLogout} />

      <LoadingOverlay visible={loading} text={loadingText} />
      <CropperModal
        visible={cropVisible}
        imageSrc={cropSrc}
        cropperReady={cropperReady}
        onApply={handleCropApply}
        onCancel={handleCropCancel}
      />

      <div className="intro-text">
        <p>Cal-FIT provides comprehensive analysis of your food labels, offering valuable insights into nutritional content and potential health risks.</p>
      </div>

      {error && <div className="scan-error-banner">{error}</div>}

      <div className="parent">
        <div className="container">
          <h2>Scan a Barcode</h2>
          <input
            ref={barcodeInputRef}
            type="file"
            accept="image/*"
            onChange={e => openCropper("barcode", e.target.files[0])}
          />
          <button onClick={() => barcodeInputRef.current?.click()}>Scan Barcode</button>
        </div>
      </div>

      <div className="text">
        <h1>How To Use Cal-FIT</h1>
      </div>
      <div className="tutorial">
        <div>
          <p>1</p>
          <h1>Scan</h1>
          <p>Scan your item using camera or uploading an image</p>
        </div>
        <div>
          <p>2</p>
          <h1>Read</h1>
          <p>Get results that highlight the good and bad in your food, backed by real nutritional science.</p>
        </div>
        <div>
          <p>3</p>
          <h1>Eat</h1>
          <p>Eat based on personalized suggestions tailored to your diet plan.</p>
        </div>
      </div>
    </div>
  );
}