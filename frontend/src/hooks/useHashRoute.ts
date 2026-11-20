import { useEffect, useState } from "react";

export interface Route {
  path: string; // e.g. "/analysis"
  segments: string[]; // ["analysis"]
  query: URLSearchParams;
}

function parse(): Route {
  const raw = window.location.hash.replace(/^#/, "") || "/";
  const [path, qs = ""] = raw.split("?");
  return { path, segments: path.split("/").filter(Boolean), query: new URLSearchParams(qs) };
}

/** Tiny hash router: no dependency needed for five pages. */
export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(parse);
  useEffect(() => {
    const onChange = () => setRoute(parse());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export const navigate = (to: string) => {
  window.location.hash = to;
};
