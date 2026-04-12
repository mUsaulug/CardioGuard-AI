import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { XaiSchema, Artifact } from "../lib/types";
import { cleanUrl } from "../lib/api";
import SanityBadge from "./SanityBadge";

interface ArtifactRendererProps {
  artifact: Artifact;
  baseUrl: string;
}

const ArtifactRenderer: React.FC<ArtifactRendererProps> = ({ artifact, baseUrl }) => {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fullUrl = cleanUrl(baseUrl, artifact.url);

  const isImage = artifact.mime.includes("image") || artifact.url.endsWith(".png");
  const isText =
    artifact.mime.includes("text") ||
    artifact.mime.includes("markdown") ||
    artifact.url.endsWith(".md");

  useEffect(() => {
    const controller = new AbortController();
    if (isText) {
      setLoading(true);
      fetch(cleanUrl(baseUrl, artifact.url), { signal: controller.signal })
        .then(res => {
          if (!res.ok) throw new Error("Failed to fetch");
          return res.text();
        })
        .then(setContent)
        .catch((err) => {
          if (err.name !== 'AbortError') {
            setContent(`Yükleme hatası: ${err.message}`);
          }
        })
        .finally(() => setLoading(false));
    }
    return () => controller.abort();
  }, [baseUrl, artifact.url, artifact.mime]);

  if (isImage) {
    return (
      <div className="flex justify-center bg-slate-50 dark:bg-slate-900 rounded border border-dashed border-slate-300 dark:border-slate-600 p-4">
        <img src={fullUrl} alt={artifact.name} className="max-w-full h-auto rounded" />
      </div>
    );
  }

  if (loading) {
    return (
      <div className="animate-pulse space-y-2 p-4">
        <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4" />
        <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded" />
        <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-5/6" />
      </div>
    );
  }

  if (content) {
    return (
      <div className="prose prose-sm prose-slate dark:prose-invert max-w-none p-4 text-slate-700 dark:text-slate-300">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    );
  }

  return <p className="text-sm text-slate-400 p-4 italic">İçerik yüklenemedi.</p>;
};

interface XaiViewerProps {
  xai: XaiSchema | null;
  baseUrl: string;
}

export default function XaiViewer({ xai, baseUrl }: XaiViewerProps) {
  if (!xai || !xai.enabled) {
    return null;
  }

  const { artifacts, run_id, sanity } = xai;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
            Klinik Rapor
          </h2>
          {run_id && (
            <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
              Çalıştırma: {run_id.substring(0, 8)}...
            </span>
          )}
        </div>
        {sanity && <SanityBadge sanity={sanity} />}
      </div>

      <div className="p-4 space-y-4 max-h-[800px] overflow-auto">
        {(!artifacts || artifacts.length === 0) ? (
          <p className="text-sm text-slate-400 dark:text-slate-500 italic text-center py-8">
            Rapor verisi bulunamadı.
          </p>
        ) : (
          artifacts.map((artifact, idx) => (
            <div
              key={artifact.url || idx}
              className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
            >
              <div className="bg-slate-100 dark:bg-slate-900 px-4 py-2 text-xs font-mono text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                <span className="font-bold">{artifact.name}</span>
                <span className="uppercase bg-slate-200 dark:bg-slate-700 px-1.5 rounded text-[10px]">
                  {artifact.type === "report_png" ? "Görsel Rapor" : artifact.type === "narrative_md" ? "Açıklama" : artifact.type}
                </span>
              </div>
              <ArtifactRenderer artifact={artifact} baseUrl={baseUrl} />
            </div>
          ))
        )}
      </div>
    </div>
  );
}
