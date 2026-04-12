import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { XaiSchema, Artifact } from "../lib/types";
import { cleanUrl, fetchTextArtifact } from "../lib/api";
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
    if (isText) {
      setLoading(true);
      fetchTextArtifact(baseUrl, artifact.url)
        .then(setContent)
        .catch((err) => setContent(`Yukleme hatasi: ${err.message}`))
        .finally(() => setLoading(false));
    }
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

  return <p className="text-sm text-slate-400 p-4 italic">Icerik yuklenemedi.</p>;
};

type TabKey = "gradcam" | "shap" | "report";

interface XaiViewerProps {
  xai: XaiSchema | null;
  baseUrl: string;
}

export default function XaiViewer({ xai, baseUrl }: XaiViewerProps) {
  const [activeTab, setActiveTab] = useState<TabKey>("gradcam");

  if (!xai || !xai.enabled) {
    return null;
  }

  const { artifacts, run_id, sanity } = xai;

  const categorize = (artifacts: Artifact[]) => {
    const gradcam: Artifact[] = [];
    const shap: Artifact[] = [];
    const report: Artifact[] = [];

    (artifacts || []).forEach((a) => {
      const name = a.name.toLowerCase();
      if (name.includes("gradcam") || name.includes("grad_cam")) {
        gradcam.push(a);
      } else if (name.includes("shap")) {
        shap.push(a);
      } else {
        report.push(a);
      }
    });

    return { gradcam, shap, report };
  };

  const categorized = categorize(artifacts);

  const tabs: { key: TabKey; label: string; count: number }[] = [
    { key: "gradcam", label: "GradCAM Haritasi", count: categorized.gradcam.length },
    { key: "shap", label: "SHAP Analizi", count: categorized.shap.length },
    { key: "report", label: "Klinik Rapor", count: categorized.report.length },
  ];

  const currentArtifacts = categorized[activeTab];

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
        <div>
          <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
            Aciklanabilir Yapay Zeka (XAI)
          </h2>
          {run_id && (
            <span className="text-[10px] text-slate-500 dark:text-slate-400 font-mono">
              Calistirma: {run_id.substring(0, 8)}...
            </span>
          )}
        </div>
        {sanity && <SanityBadge sanity={sanity} />}
      </div>

      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-4 py-2.5 text-sm font-medium transition border-b-2 ${
                activeTab === tab.key
                  ? "border-blue-500 text-blue-500 dark:text-blue-400"
                  : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
            >
              {tab.label}
              {tab.count > 0 && (
                <span className="ml-1.5 text-[10px] bg-slate-200 dark:bg-slate-700 px-1.5 py-0.5 rounded-full">
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </nav>
      </div>

      <div className="p-4 space-y-4 max-h-[600px] overflow-auto">
        {currentArtifacts.length === 0 ? (
          <p className="text-sm text-slate-400 dark:text-slate-500 italic text-center py-8">
            Bu kategoride artefakt bulunamadi.
          </p>
        ) : (
          currentArtifacts.map((artifact, idx) => (
            <div
              key={idx}
              className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
            >
              <div className="bg-slate-100 dark:bg-slate-900 px-4 py-2 text-xs font-mono text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center">
                <span className="font-bold">{artifact.name}</span>
                <span className="uppercase bg-slate-200 dark:bg-slate-700 px-1.5 rounded text-[10px]">
                  {artifact.type}
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
