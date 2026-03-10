import { useState, useEffect } from "react";
import api from "../api/axios";

const GoogleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
    <path d="M19.8055 10.2292C19.8055 9.55055 19.7501 8.86668 19.6305 8.19446H10.2002V12.0486H15.6014C15.3773 13.2905 14.6571 14.3894 13.6025 15.0875V17.5862H16.8251C18.7173 15.8444 19.8055 13.2723 19.8055 10.2292Z" fill="#4285F4"/>
    <path d="M10.2002 20C12.9593 20 15.2732 19.1056 16.8294 17.5862L13.6068 15.0875C12.7065 15.6972 11.5521 16.0431 10.2045 16.0431C7.54388 16.0431 5.29649 14.2834 4.50661 11.9167H1.1792V14.4931C2.77705 17.6681 6.30493 20 10.2002 20Z" fill="#34A853"/>
    <path d="M4.50235 11.9167C4.06702 10.6748 4.06702 9.32917 4.50235 8.08723V5.51086H1.17919C-0.395153 8.63752 -0.395153 12.3667 1.17919 15.4931L4.50235 11.9167Z" fill="#FBBC04"/>
    <path d="M10.2002 3.95834C11.6251 3.93611 13.0017 4.47223 14.0388 5.45834L16.8936 2.60278C15.1827 0.990557 12.9377 0.0847804 10.2002 0.111123C6.30493 0.111123 2.77705 2.44307 1.1792 5.61112L4.50235 8.18749C5.28797 5.81668 7.53962 3.95834 10.2002 3.95834Z" fill="#EA4335"/>
  </svg>
);

export default function Login() {
  const [tab, setTab] = useState("login");
  const [alert, setAlert] = useState(null); // { type: 'success' | 'error', message }
  const [loading, setLoading] = useState(false);

  // Login form state
  const [loginForm, setLoginForm] = useState({ username: "", password: "" });
  // Signup form state
  const [signupForm, setSignupForm] = useState({ username: "", password: "" });

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("username", loginForm.username);
      formData.append("password", loginForm.password);

      const res = await api.post("/login", formData);
      if (res.data?.redirect) {
        window.location.href = res.data.redirect;
      } else {
        window.location.href = res.data.redirect || "/dashboard"
      }
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || "Login failed. Please try again.";
      showAlert("error", msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("username", signupForm.username);
      formData.append("password", signupForm.password);

      const res = await api.post("/signup", formData);
      if (res.data?.redirect) {
        window.location.href = res.data.redirect;
      } else {
        window.location.href = res.data.redirect || "/profile-form"
      }
    } catch (err) {
      const msg = err.response?.data?.error || err.response?.data?.message || "Signup failed. Please try again.";
      showAlert("error", msg);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleLogin = () => {
    window.location.href = "http://localhost:5000/auth/google";
  };

  return (
    <div style={styles.body}>
      {/* ── Left Panel ── */}
      <div style={styles.leftPanel}>
        <div style={styles.logoSection}>
          <div style={styles.logo}>Cal-FIT</div>
        </div>

        <div style={styles.authHeader}>
          <h1 style={styles.authTitle}>
            {tab === "login" ? "Log in to your account" : "Create your account"}
          </h1>
          <p style={styles.authSubtitle}>
            {tab === "login" ? (
              <>Don't have an account?{" "}
                <span style={styles.link} onClick={() => setTab("signup")}>Sign Up</span>
              </>
            ) : (
              <>Already have an account?{" "}
                <span style={styles.link} onClick={() => setTab("login")}>Log in</span>
              </>
            )}
          </p>
        </div>

        {/* Alert */}
        {alert && (
          <div style={alert.type === "error" ? styles.alertError : styles.alertSuccess}>
            <span>{alert.type === "error" ? "⚠️" : "✓"}</span>
            {alert.message}
          </div>
        )}

        {/* OAuth */}
        <div style={styles.oauthButtons}>
          <button type="button" style={styles.oauthButton} onClick={handleGoogleLogin}
            onMouseEnter={e => Object.assign(e.currentTarget.style, styles.oauthButtonHover)}
            onMouseLeave={e => Object.assign(e.currentTarget.style, styles.oauthButton)}>
            <GoogleIcon /> Continue with Google
          </button>
        </div>

        <div style={styles.divider}>
          <span style={styles.dividerSpan}>Or with email and password</span>
        </div>

        {/* ── Login Form ── */}
        {tab === "login" && (
          <form onSubmit={handleLogin}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Username</label>
              <input
                style={styles.input}
                type="text"
                placeholder="Enter your username"
                autoComplete="username"
                required
                value={loginForm.username}
                onChange={e => setLoginForm({ ...loginForm, username: e.target.value })}
                onFocus={e => Object.assign(e.target.style, styles.inputFocus)}
                onBlur={e => Object.assign(e.target.style, styles.input)}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Password</label>
              <input
                style={styles.input}
                type="password"
                placeholder="Enter your password"
                autoComplete="current-password"
                required
                value={loginForm.password}
                onChange={e => setLoginForm({ ...loginForm, password: e.target.value })}
                onFocus={e => Object.assign(e.target.style, styles.inputFocus)}
                onBlur={e => Object.assign(e.target.style, styles.input)}
              />
            </div>
            <button type="submit" style={styles.submitButton} disabled={loading}
              onMouseEnter={e => !loading && Object.assign(e.currentTarget.style, styles.submitButtonHover)}
              onMouseLeave={e => Object.assign(e.currentTarget.style, styles.submitButton)}>
              {loading ? "Logging in..." : "Next"}
            </button>
          </form>
        )}

        {/* ── Signup Form ── */}
        {tab === "signup" && (
          <form onSubmit={handleSignup}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Username</label>
              <input
                style={styles.input}
                type="text"
                placeholder="Choose a username"
                autoComplete="username"
                required
                value={signupForm.username}
                onChange={e => setSignupForm({ ...signupForm, username: e.target.value })}
                onFocus={e => Object.assign(e.target.style, styles.inputFocus)}
                onBlur={e => Object.assign(e.target.style, styles.input)}
              />
            </div>
            <div style={styles.formGroup}>
              <label style={styles.label}>Password</label>
              <input
                style={styles.input}
                type="password"
                placeholder="Create a strong password"
                autoComplete="new-password"
                required
                value={signupForm.password}
                onChange={e => setSignupForm({ ...signupForm, password: e.target.value })}
                onFocus={e => Object.assign(e.target.style, styles.inputFocus)}
                onBlur={e => Object.assign(e.target.style, styles.input)}
              />
            </div>
            <button type="submit" style={styles.submitButton} disabled={loading}
              onMouseEnter={e => !loading && Object.assign(e.currentTarget.style, styles.submitButtonHover)}
              onMouseLeave={e => Object.assign(e.currentTarget.style, styles.submitButton)}>
              {loading ? "Creating account..." : "Create Account"}
            </button>
          </form>
        )}
      </div>

      {/* ── Right Panel ── */}
      <div style={styles.rightPanel}>
        <div style={styles.dec1} />
        <div style={styles.dec2} />
        <div style={styles.dec3} />

        <div style={styles.heroContent}>
          <h2 style={styles.heroTitle}>
            Introducing the<br />
            <span style={styles.highlight}>Cal-FIT Smart Scanner</span>
          </h2>
          <p style={styles.heroText}>
            Connect to your nutrition goals with AI-powered scanning tools.
            Analyze food labels instantly with natural language insights,
            personalized recommendations, and real-time tracking.
          </p>
          <a href="#" style={styles.heroCta}>Try now</a>

          <div style={styles.featuresList}>
            {[
              { icon: "📸", title: "Smart Scanning", desc: "Instant nutrition analysis from any food label" },
              { icon: "🎯", title: "Personalized Insights", desc: "AI-powered recommendations tailored to you" },
              { icon: "📊", title: "Track Progress", desc: "Monitor your nutrition goals in real-time" },
            ].map((f) => (
              <div key={f.title} style={styles.featureItem}>
                <div style={styles.featureIcon}>{f.icon}</div>
                <div>
                  <strong style={styles.featureTitle}>{f.title}</strong>
                  <p style={styles.featureDesc}>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Montserrat', sans-serif; }
        @keyframes float {
          0%, 100% { transform: translateY(0) scale(1); opacity: 0.3; }
          50% { transform: translateY(-30px) scale(1.1); opacity: 0.5; }
        }
        @keyframes gradientShift {
          0%, 100% { transform: translate(0,0) rotate(0deg); }
          50% { transform: translate(-5%,-5%) rotate(5deg); }
        }
        @media (max-width: 968px) {
          .right-panel { display: none !important; }
          .left-panel { width: 100% !important; min-width: auto !important; }
        }
      `}</style>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = {
  body: {
    fontFamily: "'Montserrat', sans-serif",
    minHeight: "100vh",
    display: "flex",
    overflow: "hidden",
  },
  leftPanel: {
    width: "40%",
    minWidth: 450,
    background: "#ffffff",
    display: "flex",
    flexDirection: "column",
    padding: "60px 80px",
    overflowY: "auto",
    boxShadow: "4px 0 30px rgba(0,0,0,0.1)",
    zIndex: 10,
  },
  logoSection: { marginBottom: 50 },
  logo: {
    fontSize: 32,
    fontWeight: 800,
    color: "#84BF04",
    letterSpacing: 1,
    display: "flex",
    alignItems: "center",
    gap: 10,
    borderLeft: "8px solid #84BF04",
    paddingLeft: 12,
  },
  authHeader: { marginBottom: 40 },
  authTitle: { fontSize: 32, fontWeight: 700, color: "#252601", marginBottom: 12, lineHeight: 1.2 },
  authSubtitle: { color: "#6B7280", fontSize: 15, fontWeight: 400 },
  link: { color: "#84BF04", textDecoration: "none", fontWeight: 600, cursor: "pointer" },
  alertSuccess: {
    padding: "12px 16px", borderRadius: 8, marginBottom: 20, fontSize: 14,
    display: "flex", alignItems: "center", gap: 10,
    background: "rgba(132,191,4,0.15)", color: "#252601", border: "2px solid #84BF04",
  },
  alertError: {
    padding: "12px 16px", borderRadius: 8, marginBottom: 20, fontSize: 14,
    display: "flex", alignItems: "center", gap: 10,
    background: "#FEE2E2", color: "#991B1B", border: "2px solid #EF4444",
  },
  oauthButtons: { display: "flex", flexDirection: "column", gap: 12, marginBottom: 30 },
  oauthButton: {
    width: "100%", padding: "14px 20px", border: "2px solid #E5E7EB", borderRadius: 8,
    background: "white", color: "#374151", fontSize: 15, fontWeight: 600,
    cursor: "pointer", transition: "all 0.3s ease", display: "flex",
    alignItems: "center", justifyContent: "center", gap: 12,
    fontFamily: "'Montserrat', sans-serif",
  },
  oauthButtonHover: {
    background: "#F9FAFB", borderColor: "#D1D5DB",
    transform: "translateY(-1px)", boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
  },
  divider: { textAlign: "center", margin: "30px 0", position: "relative" },
  dividerSpan: {
    background: "white", padding: "0 15px", color: "#9CA3AF",
    fontSize: 13, fontWeight: 500, position: "relative",
    zIndex: 1,
    // Line via pseudo elements not possible inline; use outline trick
    boxShadow: "0 0 0 1px #E5E7EB",
  },
  formGroup: { marginBottom: 20 },
  label: { display: "block", marginBottom: 8, color: "#374151", fontSize: 14, fontWeight: 600 },
  input: {
    width: "100%", padding: "12px 16px", border: "2px solid #E5E7EB", borderRadius: 8,
    fontSize: 15, background: "white", fontFamily: "'Montserrat', sans-serif",
    color: "#374151", outline: "none", transition: "all 0.3s ease",
  },
  inputFocus: {
    width: "100%", padding: "12px 16px", borderRadius: 8, fontSize: 15,
    background: "white", fontFamily: "'Montserrat', sans-serif", color: "#374151",
    outline: "none", border: "2px solid #84BF04", boxShadow: "0 0 0 3px rgba(132,191,4,0.1)",
  },
  submitButton: {
    width: "100%", padding: 14,
    background: "linear-gradient(135deg, #84BF04, #9dd305)",
    color: "#252601", border: "none", borderRadius: 8, fontSize: 16, fontWeight: 700,
    cursor: "pointer", transition: "all 0.3s ease", marginTop: 10,
    fontFamily: "'Montserrat', sans-serif",
    boxShadow: "0 4px 15px rgba(132,191,4,0.3)",
  },
  submitButtonHover: {
    width: "100%", padding: 14,
    background: "linear-gradient(135deg, #84BF04, #9dd305)",
    color: "#252601", border: "none", borderRadius: 8, fontSize: 16, fontWeight: 700,
    cursor: "pointer", fontFamily: "'Montserrat', sans-serif",
    transform: "translateY(-2px)", boxShadow: "0 6px 20px rgba(132,191,4,0.4)",
  },
  // Right panel
  rightPanel: {
    flex: 1,
    background: "linear-gradient(135deg, #0d5e3a 0%, #116646 100%)",
    display: "flex", alignItems: "center", justifyContent: "center",
    padding: 80, position: "relative", overflow: "hidden",
  },
  dec1: {
    position: "absolute", width: 300, height: 300, top: "10%", right: "10%",
    borderRadius: "50%", background: "rgba(132,191,4,0.1)",
    animation: "float 8s ease-in-out infinite",
  },
  dec2: {
    position: "absolute", width: 200, height: 200, bottom: "15%", left: "5%",
    borderRadius: "50%", background: "rgba(132,191,4,0.1)",
    animation: "float 8s ease-in-out 2s infinite",
  },
  dec3: {
    position: "absolute", width: 150, height: 150, top: "60%", right: "20%",
    borderRadius: "50%", background: "rgba(132,191,4,0.1)",
    animation: "float 8s ease-in-out 4s infinite",
  },
  heroContent: { maxWidth: 600, color: "white", position: "relative", zIndex: 1 },
  heroTitle: { fontSize: 48, fontWeight: 800, marginBottom: 20, lineHeight: 1.2, color: "white" },
  highlight: { color: "#84BF04" },
  heroText: { fontSize: 18, lineHeight: 1.7, marginBottom: 30, opacity: 0.95, fontWeight: 400 },
  heroCta: {
    display: "inline-flex", alignItems: "center", gap: 8,
    padding: "12px 28px", background: "rgba(255,255,255,0.15)",
    backdropFilter: "blur(10px)", border: "2px solid rgba(255,255,255,0.3)",
    borderRadius: 8, color: "white", textDecoration: "none",
    fontWeight: 600, fontSize: 16, transition: "all 0.3s ease",
  },
  featuresList: { marginTop: 40, display: "grid", gap: 16 },
  featureItem: { display: "flex", alignItems: "flex-start", gap: 12 },
  featureIcon: {
    width: 40, height: 40, background: "rgba(132,191,4,0.2)", borderRadius: 8,
    display: "flex", alignItems: "center", justifyContent: "center",
    fontSize: 20, flexShrink: 0,
  },
  featureTitle: { display: "block", fontSize: 16, marginBottom: 4, color: "white" },
  featureDesc: { fontSize: 14, opacity: 0.85, margin: 0, lineHeight: 1.5 },
};