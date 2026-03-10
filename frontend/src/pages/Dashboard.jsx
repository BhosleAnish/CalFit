import { useEffect, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";
import ScrollStack, { ScrollStackItem } from "../components/ScrollStack";

// ─── Rotating Text ────────────────────────────────────────────────────────────
function RotatingText() {
  const containerRef = useRef(null);
  const indexRef = useRef(0);
  const texts = ["Fastest", "Smartest", "Easiest", "Healthiest"];

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    function createWordElement(word) {
      const wordSpan = document.createElement("span");
      wordSpan.className = "tr-word";
      word.split("").forEach((char, i) => {
        const s = document.createElement("span");
        s.className = "tr-char";
        s.textContent = char;
        s.style.animationDelay = `${i * 0.025}s`;
        wordSpan.appendChild(s);
      });
      return wordSpan;
    }

    function animateText() {
      const current = el.querySelector(".tr-word");
      if (current) {
        const chars = current.querySelectorAll(".tr-char");
        chars.forEach((c, i) => {
          c.classList.add("exit");
          c.style.animationDelay = `${(chars.length - 1 - i) * 0.025}s`;
        });
        setTimeout(() => {
          el.innerHTML = "";
          indexRef.current = (indexRef.current + 1) % texts.length;
          el.appendChild(createWordElement(texts[indexRef.current]));
        }, 600);
      } else {
        el.appendChild(createWordElement(texts[indexRef.current]));
      }
    }

    animateText();
    const interval = setInterval(animateText, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={s.rotatingWrapper}>
      <span ref={containerRef} style={s.rotatingText} />
    </div>
  );
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

// ─── Dashboard ────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const navigate = useNavigate();

  const handleLogout = async () => {
    try { await fetch("/api/logout", { method: "POST", credentials: "include" }); } catch (_) {}
    navigate("/");
  };

  return (
    <div style={s.page}>
      <Navbar onLogout={handleLogout} />

      {/* ── Hero ── */}
      <section style={s.heroSection}>
        <div style={s.heroContent}>
          <div style={s.heroImage}>
            <img
              src="https://i.pinimg.com/1200x/b5/0b/15/b50b15ce020575f8602432fa0b7dbf6a.jpg"
              alt="Nutrition Facts"
              style={s.heroImg}
            />
          </div>
          <div style={s.heroText}>
            <h1 style={s.heroH1}>
              Make <RotatingText /> Food Choices
            </h1>
            <p style={s.heroP}>Cal-FIT helps you understand nutrition labels, one scan at a time.</p>
          </div>
        </div>
      </section>

      {/* ── Scroll Stack Cards ── */}
      <ScrollStack
        useWindowScroll={true}
        itemDistance={50}
        itemScale={0.04}
        itemStackDistance={25}
        stackPosition="15%"
        baseScale={0.88}
      >
        {/* Card 1 — Intro */}
        <ScrollStackItem itemClassName="dash-card-intro">
          <div style={s.introCard}>
            <h2 style={s.introH2}>Do you know how to read this?</h2>
            <img src="http://localhost:5000/static/images/barcodeImage_1.png" alt="Barcode" style={s.introImg} />
          </div>
        </ScrollStackItem>

        {/* Card 2 — Intro */}
        <ScrollStackItem itemClassName="dash-card-intro">
          <div style={s.introCard}>
            <h2 style={s.introH2}>Or if this is healthy for you?</h2>
            <img src="http://localhost:5000/static/images/barcodeImage_2.png" alt="Nutrition Label" style={s.introImg} />
          </div>
        </ScrollStackItem>

        {/* Card 3 — Solution */}
        <ScrollStackItem itemClassName="dash-card-solution">
          <div style={s.solutionCard}>
            <h2 style={s.solutionH2}>If not, there's no need to worry!</h2>
            <p style={s.solutionSubtitle}>CalFit scans these for you and gives insights</p>
            <p style={s.solutionCta}>Start scanning to see more</p>
          </div>
        </ScrollStackItem>

        {/* Card 4 — CTA */}
        <ScrollStackItem itemClassName="dash-card-dark">
          <div style={s.ctaCard}>
            <div style={s.ctaContent}>
              <h3 style={s.ctaH3}>Scan &amp; Analyze</h3>
              <p style={s.ctaP}>Upload a food label and instantly check for health risks.</p>
              <Link to="/scan" style={s.ctaLink}>Start Scan</Link>
            </div>
            <img src="http://localhost:5000/static/images/scan1.png" alt="Scan" style={s.ctaImg} />
          </div>
        </ScrollStackItem>
      </ScrollStack>

      {/* ── Footer ── */}
      <footer style={s.footer}>
        <div style={s.footerContent}>
          <p style={{ marginBottom: 10, fontSize: 14, color: "#D9D6D0" }}>
            &copy; 2025 Cal-FIT. All rights reserved.
          </p>
          <div style={s.socials}>
            <a href="mailto:anish22it@mes.ac.in" target="_blank" rel="noopener noreferrer" style={s.socialLink}>✉️</a>
            <a href="https://www.linkedin.com/in/anishbhosle04/" target="_blank" rel="noopener noreferrer" style={s.socialLink}>in</a>
            <a href="https://github.com/BhosleAnish" target="_blank" rel="noopener noreferrer" style={s.socialLink}>gh</a>
          </div>
        </div>
      </footer>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

        @keyframes headerEnter { to { opacity:1; transform:translateY(0); } }
        @keyframes slideInLeft  { to { opacity:1; transform:translateX(0); } }
        @keyframes slideInRight { to { opacity:1; transform:translateX(0); } }
        @keyframes trSlideUp   { from { transform:translateY(100%); opacity:0; } to { transform:translateY(0); opacity:1; } }
        @keyframes trSlideDown { from { transform:translateY(0); opacity:1; } to { transform:translateY(-120%); opacity:0; } }

        .tr-word { display:inline-flex; overflow:hidden; }
        .tr-char { display:inline-block; animation: trSlideUp 0.6s cubic-bezier(0.34,1.56,0.64,1) both; }
        .tr-char.exit { animation: trSlideDown 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards; }

        .dash-card-intro {
          background: linear-gradient(135deg, rgba(255,255,255,0.97) 0%, rgba(217,214,208,0.97) 100%) !important;
          border: 2px solid rgba(132,191,4,0.15);
        }
        .dash-card-solution {
          background: linear-gradient(135deg, #84BF04 0%, #9dd305 100%) !important;
        }
        .dash-card-dark {
          background: #0D0D0D !important;
          border: 2px solid rgba(132,191,4,0.2);
        }

        

        nav a:hover {
          color: #84BF04 !important;
          background: rgba(37,38,1,0.9) !important;
          transform: translateY(-2px) scale(1.05) !important;
        }
      `}</style>
    </div>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = {
  page: {
    fontFamily: "'Montserrat', sans-serif",
    background: "linear-gradient(135deg, #D9D6D0 0%, #c4c1bb 100%)",
    color: "#252601",
    overflowX: "hidden",
  },
  header: {
    position: "sticky", top: 0, height: 80, zIndex: 1000,
    background: "linear-gradient(135deg, #84BF04 0%, #9dd305 100%)",
    padding: "0 40px", display: "flex", justifyContent: "space-between", alignItems: "center",
    boxShadow: "0 4px 20px rgba(132,191,4,0.3)", borderBottom: "2px solid rgba(37,38,1,0.1)",
    opacity: 0, transform: "translateY(-30px)", animation: "headerEnter 1s ease forwards",
  },
  logo: { fontSize: 28, fontWeight: "bold", color: "#252601", letterSpacing: 1 },
  mainNav: {
    position: "absolute", left: "50%", transform: "translateX(-50%)",
    display: "flex", gap: 30, background: "rgba(37,38,1,0.05)",
    padding: "10px 25px", borderRadius: 50, backdropFilter: "blur(10px)",
  },
  navLink: { color: "#252601", textDecoration: "none", fontSize: 16, fontWeight: 600, padding: "10px 18px", borderRadius: 25 },
  logoutBtn: {
    background: "rgba(37,38,1,0.15)", border: "2px solid rgba(37,38,1,0.3)",
    color: "#252601", padding: "8px 20px", borderRadius: 25, fontWeight: 600,
    cursor: "pointer", fontFamily: "'Montserrat', sans-serif", fontSize: 14,
  },
  heroSection: {
    minHeight: "90vh", display: "flex", alignItems: "center",
    justifyContent: "center", padding: "40px 60px",
  },
  heroContent: {
    display: "flex", alignItems: "center", justifyContent: "space-between",
    gap: 60, maxWidth: 1400, width: "100%", margin: "0 auto",
  },
  heroImage: {
    flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
    opacity: 0, transform: "translateX(-50px)", animation: "slideInLeft 1s 0.3s ease forwards",
  },
  heroImg: { maxWidth: "100%", maxHeight: 600, objectFit: "contain", borderRadius: 20 },
  heroText: {
    flex: 1, padding: 40, color: "#252601", textAlign: "left",
    opacity: 0, transform: "translateX(50px)", animation: "slideInRight 1s 0.5s ease forwards",
  },
  heroH1: { fontSize: 65, marginBottom: 30, color: "#84BF04", lineHeight: 1.2 },
  rotatingWrapper: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    background: "linear-gradient(135deg, #84BF04 0%, #9dd305 100%)",
    padding: "8px 20px", borderRadius: 12, overflow: "hidden",
    minWidth: 200, height: 70, verticalAlign: "middle",
  },
  rotatingText: { display: "inline-flex", alignItems: "center", fontSize: 65, fontWeight: 700, color: "#252601" },
  heroP: { fontSize: 20, color: "#252601", fontWeight: 500, marginTop: 20 },

  // Card contents
  introCard: { display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 24, width: "100%", height: "100%" },
  introH2: { fontSize: 44, color: "#252601", fontWeight: 700, lineHeight: 1.3, textAlign: "center" },
  introImg: { maxHeight: 200, maxWidth: "80%", objectFit: "contain", borderRadius: 12, boxShadow: "0 8px 25px rgba(0,0,0,0.15)" },

  solutionCard: { textAlign: "center", width: "100%", height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 },
  solutionH2: { fontSize: 44, color: "#252601", fontWeight: 800 },
  solutionSubtitle: { fontSize: 26, color: "#252601", fontWeight: 600 },
  solutionCta: { fontSize: 20, color: "#252601", fontWeight: 500, opacity: 0.85 },

  ctaCard: { display: "flex", alignItems: "center", gap: 40, width: "100%", height: "100%" },
  ctaContent: { flex: 1, textAlign: "left" },
  ctaH3: { marginBottom: 15, color: "#84BF04", fontSize: 28, fontWeight: 700 },
  ctaP: { color: "#D9D6D0", marginBottom: 20, fontSize: 16, lineHeight: 1.6 },
  ctaLink: {
    background: "linear-gradient(135deg, #84BF04 0%, #9dd305 100%)",
    color: "#252601", padding: "12px 28px", borderRadius: 50,
    textDecoration: "none", fontWeight: 700, display: "inline-block",
  },
  ctaImg: { maxHeight: 140, width: "auto", objectFit: "contain", borderRadius: 12 },

  footer: {
    background: "linear-gradient(135deg, #252601 0%, #1a1b01 100%)",
    padding: "40px 20px", borderTop: "3px solid #84BF04", textAlign: "center",
    position: "relative", zIndex: 2,
  },
  footerContent: { display: "flex", flexDirection: "column", alignItems: "center", maxWidth: 1200, margin: "auto" },
  socials: { display: "flex", gap: 15, marginTop: 15 },
  socialLink: {
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    width: 40, height: 40, background: "rgba(132,191,4,0.1)", borderRadius: "50%",
    color: "#D9D6D0", textDecoration: "none", fontWeight: 700, fontSize: 14,
  },
};