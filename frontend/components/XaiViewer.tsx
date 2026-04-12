import React, { useEffect, useState } from "react";
import { XaiSchema, Artifact } from "../lib/types";
import { cleanUrl, fetchTextArtifact } from "../lib/api";

interface ArtifactRendererProps {
  artifact: Artifact;
  baseUrl: string;
}

const ArtifactRenderer: React.FC<ArtifactRendererProps> = ({ artifact, baseUrl }) => {
  if (!artifact || !artifact.url) return null;
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fullUrl = cleanUrl(baseUrl, artifact.url);

  useEffect(() => {
    if (artifact.mime.includes("text") || artifact.mime.includes("markdown") || artifact.url.endsWith(".md")) {
      setLoading(true);
      fetchTextArtifact(baseUrl, artifact.url)
        .then(setContent)
        .catch((err) => setContent(`Error loading text: ${err.message}`))
        .finally(() => setLoading(false));
    }
  }, [baseUrl, artifact.url, artifact.mime]);

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden shadow-sm bg-white">
      <div className="bg-gray-100 px-4 py-2 text-xs font-mono text-gray-600 border-b flex justify-between items-center">
        <span className="font-bold">{artifact.name}</span>
        <span className="uppercase bg-gray-200 px-1 rounded text-[10px]">{artifact.type}</span>
      </div>

      <div className="p-4 overflow-auto max-h-[500px] custom-scrollbar bg-white">
        {artifact.mime.includes("image") || artifact.url.endsWith(".png") ? (
          <div className="flex justify-center bg-gray-50 rounded border border-dashed border-gray-200 p-2">
            <img src={fullUrl} alt={artifact.name} className="max-w-full h-auto rounded" />
          </div>
        ) : (
          <div className="prose prose-sm prose-slate max-w-none">
            {loading ? (
              <div className="animate-pulse flex space-x-4">
                <div className="flex-1 space-y-2 py-1">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-200 rounded"></div>
                  <div className="h-4 bg-gray-200 rounded w-5/6"></div>
                </div>
              </div>
            ) : (
              <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700 leading-relaxed">{content || "No content loaded."}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

interface XaiViewerProps {
  xai: XaiSchema | null;
  baseUrl: string;
}

export default function XaiViewer({ xai, baseUrl }: XaiViewerProps) {
  if (!xai || !xai.enabled) {
    return (
      <div className="p-4 bg-gray-50 border border-gray-200 rounded-md text-gray-500 text-sm italic flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>
        XAI content not requested or disabled.
      </div>
    );
  }

  const { artifacts, run_id, sanity } = xai;

  return (
    <div className="space-y-6 mt-6 border-t pt-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">Explainable AI (XAI) Artifacts</h3>
        <span className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded-full font-mono">Run ID: {run_id?.substring(0, 8)}...</span>
      </div>

      {/* Sanity Check Info if available */}
      {sanity && (
        <div className="bg-yellow-50 border border-yellow-200 p-3 rounded text-sm text-yellow-800">
          <strong className="block mb-1">Sanity Check Data:</strong>
          <pre className="text-xs overflow-auto custom-scrollbar bg-white/50 p-2 rounded border border-yellow-100">{JSON.stringify(sanity, null, 2)}</pre>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6">
        {(artifacts || []).map((artifact, idx) => (
          <ArtifactRenderer key={idx} artifact={artifact} baseUrl={baseUrl} />
        ))}
      </div>
    </div>
  );
}