// Application shell: government-grade header + content area.
export default function Layout({ children }) {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <h1 className="text-xl font-semibold tracking-wide">VARUNA AI</h1>
            <p className="text-xs text-slate-400">
              Explainable AI — Extreme Rainfall Intelligence &amp; Early Warning System
            </p>
          </div>
          <span className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-400">
            SIH260006 · ISRO
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">{children}</main>
    </div>
  );
}
