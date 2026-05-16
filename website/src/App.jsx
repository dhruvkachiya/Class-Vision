import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { PerspectiveCamera } from '@react-three/drei';
import { motion, useAnimation, useInView } from 'framer-motion';
import * as THREE from 'three';

// ─── PARTICLE NEURAL NETWORK (Three.js) ────────────────────────────────────

function Particles() {
  const count = 160;
  const mesh = useRef();

  const [positions, speeds] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const spd = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 22;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 22;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 22;
      spd[i * 3] = (Math.random() - 0.5) * 0.004;
      spd[i * 3 + 1] = (Math.random() - 0.5) * 0.004;
      spd[i * 3 + 2] = (Math.random() - 0.5) * 0.004;
    }
    return [pos, spd];
  }, []);

  useFrame(() => {
    if (!mesh.current) return;
    const arr = mesh.current.geometry.attributes.position.array;
    for (let i = 0; i < count; i++) {
      arr[i * 3] += speeds[i * 3];
      arr[i * 3 + 1] += speeds[i * 3 + 1];
      arr[i * 3 + 2] += speeds[i * 3 + 2];
      if (Math.abs(arr[i * 3]) > 11) speeds[i * 3] *= -1;
      if (Math.abs(arr[i * 3 + 1]) > 11) speeds[i * 3 + 1] *= -1;
      if (Math.abs(arr[i * 3 + 2]) > 11) speeds[i * 3 + 2] *= -1;
    }
    mesh.current.geometry.attributes.position.needsUpdate = true;
    mesh.current.rotation.y += 0.0008;
    mesh.current.rotation.x += 0.0003;
  });

  return (
    <points ref={mesh}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={count}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial color="#3b82f6" size={0.08} transparent opacity={0.7} />
    </points>
  );
}

function GlowSphere() {
  const meshRef = useRef();
  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.2;
      meshRef.current.rotation.z = t * 0.1;
      meshRef.current.scale.setScalar(1 + Math.sin(t * 0.8) * 0.04);
    }
  });

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[1.8, 1]} />
      <meshStandardMaterial
        color="#3b82f6"
        wireframe
        emissive="#1d4ed8"
        emissiveIntensity={0.4}
        transparent
        opacity={0.6}
      />
    </mesh>
  );
}

function Scene() {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 0, 10]} />
      <ambientLight intensity={0.2} />
      <pointLight position={[5, 5, 5]} color="#3b82f6" intensity={2} />
      <pointLight position={[-5, -5, -5]} color="#8b5cf6" intensity={1} />
      <GlowSphere />
      <Particles />
    </>
  );
}

// ─── ANIMATED NUMBER ────────────────────────────────────────────────────────

function AnimatedNumber({ target, suffix = '' }) {
  const [value, setValue] = useState(0);
  const ref = useRef();
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const end = parseFloat(target);
    const duration = 1800;
    const step = (end / duration) * 16;
    const timer = setInterval(() => {
      start = Math.min(start + step, end);
      setValue(parseFloat(start.toFixed(1)));
      if (start >= end) clearInterval(timer);
    }, 16);
    return () => clearInterval(timer);
  }, [inView, target]);

  return <span ref={ref}>{value}{suffix}</span>;
}

// ─── SECTION REVEAL ─────────────────────────────────────────────────────────

function Reveal({ children, delay = 0, direction = 'up' }) {
  const ref = useRef();
  const inView = useInView(ref, { once: true, margin: '-80px' });
  const controls = useAnimation();

  const variants = {
    hidden: {
      opacity: 0,
      y: direction === 'up' ? 40 : direction === 'down' ? -40 : 0,
      x: direction === 'left' ? 50 : direction === 'right' ? -50 : 0,
    },
    visible: { opacity: 1, y: 0, x: 0 },
  };

  useEffect(() => {
    if (inView) controls.start('visible');
  }, [inView]);

  return (
    <motion.div
      ref={ref}
      initial="hidden"
      animate={controls}
      variants={variants}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.4, 0.25, 1] }}
    >
      {children}
    </motion.div>
  );
}

// ─── TYPEWRITER TEXT ────────────────────────────────────────────────────────

function Typewriter({ text, speed = 60 }) {
  const [displayed, setDisplayed] = useState('');
  const [i, setI] = useState(0);

  useEffect(() => {
    if (i >= text.length) return;
    const t = setTimeout(() => {
      setDisplayed(prev => prev + text[i]);
      setI(prev => prev + 1);
    }, speed);
    return () => clearTimeout(t);
  }, [i, text, speed]);

  return <span>{displayed}<span style={{ animation: 'pulse 1s infinite', opacity: i < text.length ? 1 : 0 }}>|</span></span>;
}

// ─── FEATURE DATA ────────────────────────────────────────────────────────────

const features = [
  {
    icon: '🎯',
    title: 'HAAR Cascade Detection',
    desc: 'Multi-scale sliding window algorithm instantly locates face regions with sub-16ms latency at any resolution.',
    tag: 'OpenCV',
  },
  {
    icon: '🧠',
    title: 'LBPH Recognizer',
    desc: 'Local Binary Pattern Histograms encode 50 facial samples per student into a compact binary model file.',
    tag: 'ML Model',
  },
  {
    icon: '📸',
    title: 'Auto Image Capture',
    desc: 'Camera session automatically collects variance-rich frames and terminates after the target count.',
    tag: 'Real-time',
  },
  {
    icon: '📊',
    title: 'CSV Excel Sync',
    desc: 'Per-subject attendance files are auto-created, dated, and merged into a master log for reporting.',
    tag: 'Data',
  },
  {
    icon: '🔒',
    title: 'Local Processing',
    desc: 'Zero cloud upload. All biometric data stays on-device — compliant with student privacy norms.',
    tag: 'Private',
  },
  {
    icon: '🖥️',
    title: 'Tkinter UI Dashboard',
    desc: 'Rich tabbed GUI with Stream/Semester/Subject filters, live camera feed, and attendance table view.',
    tag: 'GUI',
  },
];

// ─── MAIN APP ────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <div>
      {/* NAVBAR */}
      <nav className="navbar">
        <div className="nav-logo" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
          <div className="logo-icon">🎓</div>
          <span>CLASS<span style={{ color: '#3b82f6' }}> VISION</span></span>
        </div>
        <ul className="nav-links">
          <li><a href="#features">Technology</a></li>
          <li><a href="#workflow">Workflow</a></li>
          <li><a href="#stack">Stack</a></li>
        </ul>
        <button className="nav-cta" onClick={() => window.open('https://github.com/dhruvkachiya/Class-Vision', '_blank')}>View on GitHub</button>
      </nav>

      {/* HERO */}
      <section className="hero">
        <div className="hero-canvas">
          <Canvas>
            <Scene />
          </Canvas>
        </div>

        {/* Scan overlay decoration */}
        <div className="scan-overlay">
          <div className="scan-line" />
          <div className="corner corner-tl" />
          <div className="corner corner-tr" />
          <div className="corner corner-bl" />
          <div className="corner corner-br" />
        </div>

        <motion.div
          className="hero-badge"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <span className="pulse-dot" />
          AI-Powered Attendance System
        </motion.div>

        <motion.h1
          className="hero-heading"
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8, ease: [0.25, 0.4, 0.25, 1] }}
        >
          <span className="line-1">FACE IT.</span>
          <span className="line-2">AUTOMATE IT.</span>
        </motion.h1>

        <motion.p
          className="hero-sub"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9 }}
        >
          A Python + OpenCV system that scans the classroom in real-time,
          identifies every student, and logs attendance — zero human effort.
        </motion.p>

        <motion.div
          className="hero-actions"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
        >
          <button className="btn-primary" onClick={() => window.open('https://github.com/dhruvkachiya/Class-Vision', '_blank')}>
            View on GitHub →
          </button>
          <button className="btn-secondary" onClick={() => window.open('https://github.com/dhruvkachiya/Class-Vision#readme', '_blank')}>
            View Documentation
          </button>
        </motion.div>

        <div className="hero-scroll">
          <div className="mouse-icon"><div className="mouse-wheel" /></div>
          scroll
        </div>
      </section>

      {/* STATS ROW */}
      <div className="stats-row">
        {[
          { val: 99.2, suffix: '%', label: 'Recognition Accuracy' },
          { val: 12, suffix: 'ms', label: 'Detection Latency' },
          { val: 50, suffix: '+', label: 'Training Images/Student' },
          { val: 100, suffix: '%', label: 'Offline Processing' },
        ].map((s, i) => (
          <Reveal key={i} delay={i * 0.1}>
            <div className="stat-item">
              <span className="stat-number">
                <AnimatedNumber target={s.val} suffix={s.suffix} />
              </span>
              <span className="stat-label">{s.label}</span>
            </div>
          </Reveal>
        ))}
      </div>

      {/* FEATURES */}
      <div className="features-bg section-full" id="features">
        <div className="section">
          <Reveal>
            <div className="section-label">Core Technology</div>
            <h2 className="section-title">
              Built with precision.<br />Designed for speed.
            </h2>
            <p className="section-sub">
              Every component of the pipeline is optimized for real-world classroom conditions.
            </p>
          </Reveal>

          <div className="features-grid">
            {features.map((f, i) => (
              <Reveal key={i} delay={i * 0.08} direction="up">
                <div className="feature-card">
                  <div className="feat-icon">{f.icon}</div>
                  <div>
                    <div className="feat-title">{f.title}</div>
                    <div className="feat-desc" style={{ marginTop: 8 }}>{f.desc}</div>
                  </div>
                  <span className="feat-tag">{f.tag}</span>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>

      {/* WORKFLOW */}
      <div className="workflow-section" id="workflow">
        <div className="workflow-split">
          {/* Steps */}
          <Reveal direction="right">
            <div className="section-label">Implementation</div>
            <h2 className="section-title" style={{ marginBottom: 40 }}>
              How it all<br />comes together.
            </h2>
            <div className="steps-list">
              {[
                { n: '01', title: 'Register Students', body: 'Capture 50+ facial frames with automatic lighting normalization and store in TrainingImage/ directory.' },
                { n: '02', title: 'Train the Model', body: 'LBPH Recognizer processes all images and serializes the model to a .yml file for instant reloading.' },
                { n: '03', title: 'Run Auto Attendance', body: 'Live webcam feed is scanned frame-by-frame. Recognized faces are matched against the model with >95% confidence threshold.' },
                { n: '04', title: 'Export & Review', body: 'Attendance auto-logs to subject-specific CSVs. View reports via the built-in tabular dashboard with filters.' },
              ].map((s, i) => (
                <Reveal key={i} delay={i * 0.12} direction="right">
                  <div className="step-item">
                    <div className="step-num">{s.n}</div>
                    <div className="step-body">
                      <h4>{s.title}</h4>
                      <p>{s.body}</p>
                    </div>
                  </div>
                </Reveal>
              ))}
            </div>
          </Reveal>

          {/* Terminal */}
          <Reveal direction="left" delay={0.2}>
            <div className="terminal-card">
              <div className="terminal-topbar">
                <div className="dot-red" />
                <div className="dot-yellow" />
                <div className="dot-green" />
                <div className="terminal-title">attendance.py</div>
              </div>
              <div className="terminal-body">
                <div className="t-prompt">
                  <Typewriter text="$ python attendance.py" speed={45} />
                </div>
                <div className="t-info">[INFO] Camera feed initialized — OK</div>
                <div className="t-info">[INFO] Model loaded: trainer.yml</div>
                <div className="t-log">[SCAN] Processing frame...</div>
                <div className="t-log">[SCAN] Face region detected at (124, 80)</div>

                <div className="t-match-card">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <div className="t-avatar">👤</div>
                    <div>
                      <div className="t-match-name">Rahul Patel</div>
                      <div className="t-match-id">ID: 4884 · Confidence: 97.4%</div>
                    </div>
                  </div>
                  <div className="t-logged">Logged</div>
                </div>

                <div className="t-metrics">
                  <div className="t-metric">
                    <span className="t-metric-val">99.2%</span>
                    <span className="t-metric-name">Accuracy</span>
                  </div>
                  <div className="t-metric">
                    <span className="t-metric-val">12ms</span>
                    <span className="t-metric-name">Latency</span>
                  </div>
                  <div className="t-metric">
                    <span className="t-metric-val">8</span>
                    <span className="t-metric-name">Present</span>
                  </div>
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </div>

      {/* TECH STACK */}
      <div className="tech-section" id="stack">
        <Reveal>
          <div className="section-label">Technical Architecture</div>
          <h2 className="section-title">The Power Behind <span style={{
            background: 'linear-gradient(90deg, #3b82f6, #22d3ee)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>CLASS VISION</span></h2>
          <p className="section-sub" style={{ margin: '0 auto 60px' }}>
            A carefully selected stack for performance, reliability, and local processing.
          </p>
        </Reveal>

        <div className="tech-grid">
          {[
            { icon: '🐍', name: 'Python 3.9+', desc: 'Core logic & automation engine' },
            { icon: '👁', name: 'OpenCV', desc: 'Real-time vision processing' },
            { icon: '🧠', name: 'LBPH Recognizer', desc: 'Accurate face matching model' },
            { icon: '🖥️', name: 'Tkinter GUI', desc: 'Cross-platform dashboard' },
            { icon: '📊', name: 'Pandas', desc: 'Data logging & CSV sync' },
            { icon: '🔊', name: 'pyttsx3', desc: 'Voice feedback activation' },
            { icon: '🗂️', name: 'NumPy', desc: 'Matrix pixel calculations' },
            { icon: '📷', name: 'Haar Cascades', desc: 'Instant face detection' },
          ].map((t, i) => (
            <Reveal key={i} delay={i * 0.05} direction="up">
              <div className="tech-card">
                <div className="tech-icon-box">{t.icon}</div>
                <div>
                  <div className="tech-name">{t.name}</div>
                  <div className="tech-desc">{t.desc}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>

      {/* CTA */}
      <div className="cta-section">
        <div className="cta-glow" />
        <Reveal>
          <h2 className="cta-title">
            Stop calling rolls.<br />
            <span style={{
              background: 'linear-gradient(90deg, #3b82f6, #22d3ee)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Start scanning faces.
            </span>
          </h2>
          <p className="cta-sub">
            Clone the repo, install dependencies with<br />
            <code style={{ fontFamily: 'Space Mono', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 6 }}>pip install -r requirements.txt</code>,<br />
            and run <code style={{ fontFamily: 'Space Mono', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 6 }}>python attendance.py</code>.
          </p>
          <div className="cta-buttons">
            <button className="btn-primary" onClick={() => window.open('https://github.com/dhruvkachiya/Class-Vision', '_blank')}>Get the Code</button>
            <button className="btn-secondary" onClick={() => window.open('https://github.com/dhruvkachiya/Class-Vision#readme', '_blank')}>Read the Docs</button>
          </div>
        </Reveal>
      </div>

      {/* FOOTER */}
      <footer className="footer">
        <span>© 2026 Class Vision — All processing is local.</span>
        <span style={{ color: '#3b82f6' }}>Made with Python + OpenCV</span>
      </footer>
    </div>
  );
}
