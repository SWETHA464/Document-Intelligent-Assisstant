import Layout from "./components/Layout";
import { ToastProvider } from "./hooks/useToast";
import { useHashRoute } from "./hooks/useHashRoute";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Analysis from "./pages/Analysis";
import History from "./pages/History";
import Settings from "./pages/Settings";
import type { Operation } from "./types";

const OPS = ["qa", "compare", "extract", "summarize"];

export default function App() {
  const route = useHashRoute();
  const [first, second] = route.segments;

  let page;
  switch (first) {
    case undefined: page = <Dashboard />; break;
    case "documents": page = <Documents />; break;
    case "analysis": {
      const op = route.query.get("op");
      const docs = (route.query.get("docs") ?? "").split(",").map(Number).filter((n) => Number.isInteger(n) && n > 0);
      page = <Analysis initialOp={op && OPS.includes(op) ? (op as Operation) : null} initialDocs={docs} />;
      break;
    }
    case "history": page = <History openId={second && Number.isInteger(Number(second)) ? Number(second) : null} />; break;
    case "settings": page = <Settings />; break;
    default: page = <div className="card"><h2>Page not found</h2><a href="#/">Back to dashboard</a></div>;
  }
  return (
    <ToastProvider>
      <Layout path={route.path}>{page}</Layout>
    </ToastProvider>
  );
}
