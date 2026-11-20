import type { ReactNode } from "react";

const NAV = [
  { to: "/", label: "Dashboard", icon: "◧" },
  { to: "/documents", label: "Documents", icon: "▤" },
  { to: "/analysis", label: "Analysis", icon: "✦" },
  { to: "/history", label: "History", icon: "↺" },
  { to: "/settings", label: "Settings", icon: "⚙" },
];

export default function Layout({ path, children }: { path: string; children: ReactNode }) {
  const active = (to: string) => (to === "/" ? path === "/" : path.startsWith(to));
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">D</span>
          <div>
            <strong>DocIntel</strong>
            <small>Document Intelligence</small>
          </div>
        </div>
        <nav aria-label="Main">
          {NAV.map((n) => (
            <a key={n.to} href={`#${n.to}`} className={active(n.to) ? "nav-link active" : "nav-link"} aria-current={active(n.to) ? "page" : undefined}>
              <span aria-hidden>{n.icon}</span> {n.label}
            </a>
          ))}
        </nav>
        <p className="sidebar-note">Runs locally. Your documents stay on this machine.</p>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
