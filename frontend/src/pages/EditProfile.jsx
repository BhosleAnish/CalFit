import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";

const CONDITIONS = [
  'None','Diabetes','Hypertension','Heart Disease','Asthma','Arthritis','Thyroid Disorder','Obesity',
  'High Cholesterol','Anemia','Anxiety','Depression','Kidney Disease','Liver Disease','Gastritis','IBS',
  'PCOS','Migraine','Epilepsy','Cancer','Celiac Disease','COPD','Osteoporosis','Psoriasis','Eczema',
  'Sleep Apnea','Gout','GERD','Ulcer','Scoliosis','Lupus','Hepatitis','HIV','Chronic Pain','Back Pain',
  'Panic Disorder','Stroke','Autism','ADHD','Endometriosis','Fibromyalgia','Hyperthyroidism',
  'Hypothyroidism','Vertigo','Fatty Liver','Bronchitis','Sinusitis','Hemophilia','Tuberculosis',
  'Chronic Fatigue Syndrome'
];

const ALLERGIES = [
  'None','Peanuts','Tree Nuts','Milk','Eggs','Wheat','Soy','Fish','Shellfish','Corn','Sesame','Gluten',
  'Mustard','Celery','Lupin','Sulfites','Latex','Pollen','Dust Mites','Animal Dander','Mold','Chocolate',
  'Bananas','Kiwi','Strawberries','Apples','Tomatoes','Coconut','Garlic','Onion','Citrus Fruits','Beef',
  'Pork','Chicken','Alcohol','Caffeine','Preservatives','Artificial Colors','MSG','Honey','Berries',
  'Avocado','Spinach','Carrots','Peas','Soy Lecithin','Oats','Rice','Chili Peppers','Corn Syrup',
  'Gelatin','Sunflower Seeds'
];

const styles = `
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600;700&display=swap');

  .ep-root {
    min-height: 100vh;
    background: #f4f1eb;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a0e;
  }

  .ep-header {
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

  .ep-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 32px;
    color: #1a1a0e;
    letter-spacing: 3px;
  }

  .ep-nav { display: flex; gap: 8px; }
  .ep-nav a {
    padding: 8px 20px;
    border-radius: 50px;
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    color: #1a1a0e;
    transition: all 0.25s ease;
  }
  .ep-nav a:hover, .ep-nav a.active {
    background: #1a1a0e;
    color: #84BF04;
  }

  .ep-body {
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 24px 80px;
  }

  .ep-back {
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
    font-family: 'DM Sans', sans-serif;
  }
  .ep-back:hover { background: #84BF04; transform: translateX(-4px); }

  .ep-hero {
    background: #1a1a0e;
    border-radius: 24px;
    padding: 48px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
  }
  .ep-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(132,191,4,0.18) 0%, transparent 70%);
    pointer-events: none;
  }
  .ep-hero h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 64px;
    color: #84BF04;
    letter-spacing: 4px;
    margin: 0 0 8px;
    line-height: 1;
  }
  .ep-hero p { color: #a8a89a; font-size: 16px; margin: 0; }

  .ep-card {
    background: white;
    border-radius: 24px;
    padding: 40px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
  }

  .ep-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }

  .ep-group { display: flex; flex-direction: column; gap: 8px; }
  .ep-group.full { grid-column: 1 / -1; }

  .ep-label {
    font-size: 13px;
    font-weight: 700;
    color: #6b6b5a;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .ep-input, .ep-select {
    width: 100%;
    padding: 13px 18px;
    border: 2px solid #e8e4dc;
    border-radius: 12px;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
    color: #1a1a0e;
    background: #f9f7f3;
    transition: all 0.2s ease;
    outline: none;
    appearance: none;
  }
  .ep-input:focus, .ep-select:focus {
    border-color: #84BF04;
    background: white;
    box-shadow: 0 0 0 4px rgba(132,191,4,0.12);
  }

  /* Multi-select tag input */
  .ep-tags-wrap {
    border: 2px solid #e8e4dc;
    border-radius: 12px;
    background: #f9f7f3;
    padding: 10px 12px;
    min-height: 52px;
    cursor: text;
    transition: all 0.2s ease;
    position: relative;
  }
  .ep-tags-wrap.focused {
    border-color: #84BF04;
    background: white;
    box-shadow: 0 0 0 4px rgba(132,191,4,0.12);
  }
  .ep-tags-inner {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }
  .ep-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, #84BF04, #9dd305);
    color: #1a1a0e;
    padding: 5px 12px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 600;
    animation: tagIn 0.18s ease;
  }
  @keyframes tagIn {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
  }
  .ep-tag-x {
    cursor: pointer;
    font-size: 15px;
    line-height: 1;
    opacity: 0.7;
    background: none;
    border: none;
    padding: 0;
    color: #1a1a0e;
    font-family: inherit;
  }
  .ep-tag-x:hover { opacity: 1; }
  .ep-tag-input {
    border: none;
    outline: none;
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    color: #1a1a0e;
    min-width: 120px;
    flex: 1;
  }
  .ep-tag-input::placeholder { color: #aaa; }

  .ep-dropdown {
    position: absolute;
    top: calc(100% + 6px);
    left: 0; right: 0;
    background: white;
    border: 2px solid rgba(132,191,4,0.3);
    border-radius: 12px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.12);
    max-height: 220px;
    overflow-y: auto;
    z-index: 200;
  }
  .ep-dropdown::-webkit-scrollbar { width: 4px; }
  .ep-dropdown::-webkit-scrollbar-thumb { background: #84BF04; border-radius: 4px; }

  .ep-opt {
    padding: 10px 16px;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.15s;
    color: #1a1a0e;
  }
  .ep-opt:hover { background: rgba(132,191,4,0.12); }
  .ep-opt.selected { background: rgba(132,191,4,0.18); font-weight: 600; }
  .ep-opt-empty { padding: 12px 16px; color: #aaa; font-size: 14px; }

  .ep-submit {
    width: 100%;
    margin-top: 36px;
    padding: 16px;
    background: linear-gradient(135deg, #84BF04, #9dd305);
    color: #1a1a0e;
    border: none;
    border-radius: 50px;
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.25s ease;
    box-shadow: 0 6px 22px rgba(132,191,4,0.4);
  }
  .ep-submit:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(132,191,4,0.55);
  }
  .ep-submit:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

  .ep-toast {
    position: fixed;
    bottom: 32px; right: 32px;
    padding: 14px 24px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 14px;
    font-family: 'DM Sans', sans-serif;
    box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    z-index: 9999;
    animation: toastIn 0.3s ease;
  }
  @keyframes toastIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .ep-toast.success { background: #1a1a0e; color: #84BF04; }
  .ep-toast.error   { background: #ff5252; color: white; }

  @media (max-width: 640px) {
    .ep-grid { grid-template-columns: 1fr; }
    .ep-hero h1 { font-size: 44px; }
    .ep-header { padding: 0 16px; }
    .ep-body { padding: 24px 16px 60px; }
    .ep-card { padding: 24px; }
  }
`;

// ── Reusable multi-select tag component ──────────────────────────────────────
function TagSelect({ options, value, onChange, placeholder }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [focused, setFocused] = useState(false);
  const wrapRef = useRef(null);
  const inputRef = useRef(null);

  const filtered = options.filter(o =>
    o.toLowerCase().includes(query.toLowerCase()) && !value.includes(o)
  );

  useEffect(() => {
    const handler = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false); setFocused(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const add = (opt) => {
    onChange([...value.filter(v => v !== 'None'), opt].filter((v, i, a) => a.indexOf(v) === i));
    setQuery(""); inputRef.current?.focus();
    if (opt === 'None') onChange(['None']);
  };

  const remove = (opt) => onChange(value.filter(v => v !== opt));

  return (
    <div
      className={`ep-tags-wrap${focused ? " focused" : ""}`}
      ref={wrapRef}
      onClick={() => { setOpen(true); setFocused(true); inputRef.current?.focus(); }}
    >
      <div className="ep-tags-inner">
        {value.map(v => (
          <span className="ep-tag" key={v}>
            {v}
            <button className="ep-tag-x" onClick={e => { e.stopPropagation(); remove(v); }}>×</button>
          </span>
        ))}
        <input
          ref={inputRef}
          className="ep-tag-input"
          placeholder={value.length === 0 ? placeholder : ""}
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => { setOpen(true); setFocused(true); }}
        />
      </div>
      {open && (
        <div className="ep-dropdown">
          {filtered.length === 0
            ? <div className="ep-opt-empty">No results found</div>
            : filtered.map(o => (
              <div
                key={o}
                className={`ep-opt${value.includes(o) ? " selected" : ""}`}
                onMouseDown={e => { e.preventDefault(); add(o); }}
              >{o}</div>
            ))
          }
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function EditProfile() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState(null);

  const [form, setForm] = useState({
    full_name: "", age: "", gender: "Male",
    height_cm: "", weight_kg: "", activity_level: "sedentary",
    medical_conditions: [], allergies: []
  });

  // Load existing profile
  useEffect(() => {
    fetch("/api/profile", { credentials: "include" })
      .then(r => r.json())
      .then(data => {
        if (data.profile) {
          const p = data.profile;
          setForm({
            full_name: p.full_name || "",
            age: p.age || "",
            gender: p.gender || "Male",
            height_cm: p.height_cm || "",
            weight_kg: p.weight_kg || "",
            activity_level: p.activity_level || "sedentary",
            medical_conditions: p.medical_conditions
              ? (typeof p.medical_conditions === "string"
                  ? p.medical_conditions.split(",").map(s => s.trim()).filter(Boolean)
                  : p.medical_conditions)
              : [],
            allergies: p.allergies
              ? (typeof p.allergies === "string"
                  ? p.allergies.split(",").map(s => s.trim()).filter(Boolean)
                  : p.allergies)
              : []
          });
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const showToast = (msg, type) => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handleSubmit = async () => {
    setSaving(true);
    try {
      const payload = {
        ...form,
        medical_conditions: form.medical_conditions.join(", ") || "None",
        allergies: form.allergies.join(", ") || "None"
      };
      const res = await fetch("/api/profile", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.success) {
        showToast("✅ Profile updated successfully!", "success");
        setTimeout(() => navigate("/profile"), 1500);
      } else {
        showToast(data.error || "Failed to save profile.", "error");
      }
    } catch {
      showToast("Network error. Please try again.", "error");
    } finally {
      setSaving(false);
    }
  };

  if (loading) return (
    <>
      <style>{styles}</style>
      <div className="ep-root">
        <header className="ep-header">
          <div className="ep-logo">Cal-FIT</div>
        </header>
        <div style={{ textAlign: "center", padding: "80px", fontFamily: "'Bebas Neue', sans-serif", fontSize: "28px", letterSpacing: "3px", color: "#6b6b5a" }}>
          Loading profile…
        </div>
      </div>
    </>
  );

  return (
    <>
      <style>{styles}</style>
      <div className="ep-root">
        {/* Header */}
        <header className="ep-header">
          <div className="ep-logo">Cal-FIT</div>
          <nav className="ep-nav">
            <a href="/dashboard">Home</a>
            <a href="/scan">Scan</a>
            <a href="/profile" className="active">Profile</a>
          </nav>
        </header>

        <div className="ep-body">
          <button className="ep-back" onClick={() => navigate("/profile")}>← Back to Profile</button>

          {/* Hero */}
          <div className="ep-hero">
            <h1>Edit Profile</h1>
            <p>Keep your health details up to date for better insights</p>
          </div>

          {/* Form Card */}
          <div className="ep-card">
            <div className="ep-grid">

              <div className="ep-group">
                <label className="ep-label">Full Name</label>
                <input className="ep-input" type="text" placeholder="John Doe"
                  value={form.full_name} onChange={e => set("full_name", e.target.value)} />
              </div>

              <div className="ep-group">
                <label className="ep-label">Age</label>
                <input className="ep-input" type="number" placeholder="25" min="1"
                  value={form.age} onChange={e => set("age", e.target.value.replace(/[^0-9]/g, ""))} />
              </div>

              <div className="ep-group">
                <label className="ep-label">Gender</label>
                <select className="ep-select" value={form.gender} onChange={e => set("gender", e.target.value)}>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>

              <div className="ep-group">
                <label className="ep-label">Activity Level</label>
                <select className="ep-select" value={form.activity_level} onChange={e => set("activity_level", e.target.value)}>
                  <option value="sedentary">Sedentary</option>
                  <option value="moderate">Moderate</option>
                  <option value="active">Active</option>
                </select>
              </div>

              <div className="ep-group">
                <label className="ep-label">Height (cm)</label>
                <input className="ep-input" type="number" placeholder="170"
                  value={form.height_cm} onChange={e => set("height_cm", e.target.value)} />
              </div>

              <div className="ep-group">
                <label className="ep-label">Weight (kg)</label>
                <input className="ep-input" type="number" placeholder="70"
                  value={form.weight_kg} onChange={e => set("weight_kg", e.target.value)} />
              </div>

              <div className="ep-group full">
                <label className="ep-label">Medical Conditions</label>
                <TagSelect
                  options={CONDITIONS}
                  value={form.medical_conditions}
                  onChange={v => set("medical_conditions", v)}
                  placeholder="Search and select conditions…"
                />
              </div>

              <div className="ep-group full">
                <label className="ep-label">Allergies</label>
                <TagSelect
                  options={ALLERGIES}
                  value={form.allergies}
                  onChange={v => set("allergies", v)}
                  placeholder="Search and select allergies…"
                />
              </div>

            </div>

            <button className="ep-submit" onClick={handleSubmit} disabled={saving}>
              {saving ? "Saving…" : "Update Profile"}
            </button>
          </div>
        </div>

        {/* Toast */}
        {toast && (
          <div className={`ep-toast ${toast.type}`}>{toast.msg}</div>
        )}
      </div>
    </>
  );
}