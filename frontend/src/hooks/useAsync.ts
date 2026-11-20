import { useCallback, useEffect, useRef, useState } from "react";

/** Load data on mount (and on `reload`). Keeps the last value while reloading. */
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true; // re-set on mount: React StrictMode mounts, unmounts and remounts in development
    return () => { alive.current = false; };
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      const value = await fn();
      if (alive.current) { setData(value); setError(null); }
    } catch (e) {
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => { void reload(); }, [reload]);
  return { data, error, loading, reload, setData };
}
