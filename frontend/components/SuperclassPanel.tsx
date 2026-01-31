import React, { useState, ChangeEvent } from "react";
import { apiRequest } from "../lib/api";
import { SuperclassResponse } from "../lib/types";
import XaiViewer from "./XaiViewer";

interface Props {
  baseUrl: string;
  disabled: boolean;
}

export default function SuperclassPanel({ baseUrl, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [ensembleWeight, setEnsembleWeight] = useState(0.5);
  const [explain, setExplain] = useState(false);
  const [sanityCheck, setSanityCheck] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SuperclassResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
        alert("File size exceeds 10MB limit defined in contract.");
        e.target.value = ""; // Reset input
        setFile(null);
        return;
      }
      setFile(selectedFile);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const params = new URLSearchParams({
      ensemble_weight: ensembleWeight.toString(),
      explain: explain.toString(),
      sanity_check: (explain && sanityCheck).toString(),
    });

    try {
      // Increase timeout for prediction (e.g., 60s) as inference might take time
      const data = await apiRequest<SuperclassResponse>(
        baseUrl,
        `/predict/superclass?${params.toString()}`,
        { method: "POST", body: formData },
        60000 
      );
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-lg overflow-hidden border border-gray-100 flex flex-col h-full">
      <div className="bg-blue-600 text-white px-4 py-3 flex justify-between items-center">
        <div>
            <h2 className="font-bold text-lg">Predict Superclass</h2>
            <p className="text-blue-100 text-xs">Multilabel Classification</p>
        </div>
        <div className="bg-blue-700 p-1.5 rounded">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        </div>
      </div>

      <div className="p-4 space-y-4 flex-grow flex flex-col">
        {/* Controls */}
        <div className="space-y-4 p-4 bg-slate-50 rounded border border-slate-200">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-1">Input ECG (.npy/.npz)</label>
            <input
              type="file"
              accept=".npy,.npz"
              onChange={handleFileChange}
              disabled={disabled}
              className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 disabled:opacity-50 cursor-pointer"
            />
            <p className="text-[10px] text-gray-400 mt-1 text-right">Max size: 10MB</p>
          </div>
          
          <div>
            <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-gray-700">Ensemble Weight</label>
                <span className="text-xs font-mono bg-white px-2 py-0.5 rounded border">{ensembleWeight}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={ensembleWeight}
              onChange={(e) => setEnsembleWeight(parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full accent-blue-600 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-gray-400 px-1">
                <span>CNN</span>
                <span>Ensemble</span>
                <span>XGB</span>
            </div>
          </div>

          <div className="flex gap-6 border-t border-slate-200 pt-3">
            <label className="flex items-center space-x-2 text-sm text-gray-700 cursor-pointer">
              <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} disabled={disabled} className="rounded text-blue-600 focus:ring-blue-500" />
              <span className="font-medium">Explain (XAI)</span>
            </label>
            <label className="flex items-center space-x-2 text-sm text-gray-700 cursor-pointer">
              <input type="checkbox" checked={sanityCheck} onChange={(e) => setSanityCheck(e.target.checked)} disabled={disabled || !explain} className="rounded text-blue-600 focus:ring-blue-500 disabled:text-gray-400" />
              <span className={!explain ? 'text-gray-400' : 'font-medium'}>Sanity Check</span>
            </label>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || disabled || loading}
            className="w-full bg-blue-600 text-white py-2.5 rounded shadow-sm font-semibold hover:bg-blue-700 disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed transition flex justify-center items-center gap-2"
          >
            {loading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
            {loading ? "Processing..." : "Run Prediction"}
          </button>
        </div>

        {/* Error */}
        {error && <div className="text-red-600 text-sm bg-red-50 p-3 rounded border border-red-200 flex items-start gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 flex-shrink-0"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <span className="break-all">{error}</span>
        </div>}

        {/* Results */}
        {result && (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
            {/* Primary & Labels */}
            <div className="flex justify-between items-start bg-gradient-to-br from-white to-blue-50 p-4 rounded border shadow-sm">
              <div>
                <span className="text-[10px] text-gray-500 block uppercase tracking-wide font-bold mb-1">Primary Diagnosis</span>
                <div className="flex items-baseline gap-2">
                    <span className="text-2xl font-extrabold text-gray-900">{result.primary.label}</span>
                    <span className={`text-sm font-bold px-2 py-0.5 rounded-full ${result.primary.confidence > 0.8 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                        {(result.primary.confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="text-xs text-gray-500 mt-1 font-mono">Rule: {result.primary.rule}</div>
              </div>
              <div className="text-right max-w-[50%]">
                <span className="text-[10px] text-gray-500 block uppercase tracking-wide font-bold mb-2">Predicted Labels</span>
                <div className="flex flex-wrap gap-1 justify-end">
                  {result.predicted_labels.map(l => (
                    <span key={l} className="bg-blue-600 text-white text-xs px-2.5 py-1 rounded shadow-sm font-medium">{l}</span>
                  ))}
                  {result.predicted_labels.length === 0 && <span className="text-gray-400 text-xs italic">No findings</span>}
                </div>
              </div>
            </div>

            {/* Probabilities Table */}
            <div className="overflow-x-auto rounded border">
              <table className="min-w-full text-xs text-right">
                <thead>
                  <tr className="border-b bg-gray-100 text-gray-600 uppercase tracking-wider">
                    <th className="text-left py-2 px-3">Class</th>
                    <th className="py-2 px-3">Threshold</th>
                    <th className="py-2 px-3 bg-blue-50/50 text-blue-800">Ensemble</th>
                    <th className="py-2 px-3 text-gray-500 font-normal">CNN</th>
                    <th className="py-2 px-3 text-gray-500 font-normal">XGB</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {Object.keys(result.probabilities).map((key) => {
                    const k = key as keyof typeof result.probabilities;
                    const val = result.probabilities[k];
                    const thresh = result.thresholds[k as keyof typeof result.thresholds] || 0.5;
                    const isHigh = val > thresh;
                    return (
                      <tr key={key} className={isHigh ? "bg-blue-50/30" : "hover:bg-gray-50"}>
                        <td className="text-left font-bold py-2 px-3 text-gray-700">{key}</td>
                        <td className="py-2 px-3 text-gray-500 font-mono">{thresh.toFixed(2)}</td>
                        <td className={`py-2 px-3 font-mono ${isHigh ? 'font-bold text-blue-700' : 'text-gray-600'}`}>{val.toFixed(3)}</td>
                        <td className="py-2 px-3 text-gray-400 font-mono">{(result.sources.cnn[k]).toFixed(3)}</td>
                        <td className="py-2 px-3 text-gray-400 font-mono">{result.sources.xgb ? result.sources.xgb[k].toFixed(3) : '-'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            
            {/* Version Info */}
            <div className="text-[10px] text-gray-400 bg-gray-50 p-2 rounded border border-gray-100 font-mono flex justify-between">
              <span>v{result.versions.api_version}</span>
              <span>Hash: {result.versions.model_hash.substring(0,8)}</span>
            </div>

            {/* XAI Section */}
            <XaiViewer xai={result.xai} baseUrl={baseUrl} />
          </div>
        )}
      </div>
    </div>
  );
}