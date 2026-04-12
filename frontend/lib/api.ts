export const cleanUrl = (base: string, path: string) => {
  const cleanBase = base.replace(/\/+$/, '');
  const cleanPath = path.replace(/^\/+/, '');
  return `${cleanBase}/${cleanPath}`;
};

export async function apiRequest<T>(
  baseUrl: string,
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = 30000 // Default 30s timeout
): Promise<T> {
  const url = cleanUrl(baseUrl, endpoint);
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);

    if (!res.ok) {
      const errorText = await res.text();
      // Try to parse JSON error if possible
      try {
        const jsonError = JSON.parse(errorText);
        throw new Error(jsonError.detail || `API Error (${res.status})`);
      } catch (e) {
        throw new Error(`API Error (${res.status}): ${errorText}`);
      }
    }

    return await res.json();
  } catch (err: any) {
    clearTimeout(id);
    if (err.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs}ms`);
    }
    throw err;
  }
}

export const fetchTextArtifact = async (baseUrl: string, artifactUrl: string): Promise<string> => {
  const url = cleanUrl(baseUrl, artifactUrl);
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch artifact text");
  return res.text();
};