import React, { useState, ChangeEvent } from "react";
import { apiRequest } from "../lib/api";
import { LocalizationResponse } from "../lib/types";
import XaiViewer from "./XaiViewer";

interface Props {
  baseUrl: string;
  disabled: boolean;
}

export default function LocalizationPanel({ baseUrl, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [explain, setExplain] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<LocalizationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.size > 10 * 1024 * 1024) { // 10MB limit
        alert("File size exceeds 10MB limit defined in contract.");
        e.target.value = "";
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
      threshold: threshold.toString(),
      explain: explain.toString(),
    });

    try {
      const data = await apiRequest<LocalizationResponse>(
        baseUrl,
        `/predict/mi-localization?${params.toString()}`,
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
      <div className="bg-teal-600 text-white px-4 py-3 flex justify-between items-center">
        <div>
            <h2 className="font-bold text-lg">MI Localization</h2>
            <p className="text-teal-100 text-xs">Anatomical Region Detection</p>
        </div>
        <div className="bg-teal-700 p-1.5 rounded">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
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
              className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-teal-600 file:text-white hover:file:bg-teal-700 disabled:opacity-50 cursor-pointer"
            />
            <p className="text-[10px] text-gray-400 mt-1 text-right">Max size: 10MB</p>
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-gray-700">Threshold</label>
                <span className="text-xs font-mono bg-white px-2 py-0.5 rounded border">{threshold}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              disabled={disabled}
              className="w-full accent-teal-600 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div className="border-t border-slate-200 pt-3">
             <label className="flex items-center space-x-2 text-sm text-gray-700 cursor-pointer">
                <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} disabled={disabled} className="rounded text-teal-600 focus:ring-teal-500" />
                <span className="font-medium">Explain (XAI)</span>
             </label>
          </div>

          <button
            onClick={handleSubmit}
            disabled={!file || disabled || loading}
            className="w-full bg-teal-600 text-white py-2.5 rounded shadow-sm font-semibold hover:bg-teal-700 disabled:bg-gray-300 disabled:shadow-none disabled:cursor-not-allowed transition flex justify-center items-center gap-2"
          >
            {loading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
            {loading ? "Locate Regions" : "Run Localization"}
          </button>
        </div>

        {error && <div className="text-red-600 text-sm bg-red-50 p-3 rounded border border-red-200 flex items-start gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 flex-shrink-0"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            <span className="break-all">{error}</span>
        </div>}

        {result && (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
            
            {/* Summary */}
            <div className="bg-white p-3 rounded border shadow-sm flex flex-col gap-2">
              <div className="flex justify-between items-center border-b border-gray-100 pb-2">
                 <span className="text-[10px] text-gray-500 uppercase font-bold">MI Status</span>
                 {result.mi_detected ? (
                    <span className="bg-red-100 text-red-700 text-xs px-2 py-1 rounded font-bold animate-pulse">MI DETECTED</span>
                 ) : (
                    <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded font-bold">NO MI DETECTED</span>
                 )}
              </div>
              <div>
                <span className="text-[10px] text-gray-500 uppercase font-bold block mb-1">Affected Regions</span>
                <div className="flex flex-wrap gap-1">
                  {result.regions.map(r => (
                    <span key={r} className="bg-red-50 text-red-700 border border-red-200 text-xs px-2 py-1 rounded font-medium">{r}</span>
                  ))}
                  {result.regions.length === 0 && <span className="text-gray-400 text-xs italic">No specific regions identified</span>}
                </div>
              </div>
            </div>

            {/* Regions Visualization */}
            <div className="border rounded overflow-hidden">
                <table className="min-w-full text-xs border-collapse">
                    <thead className="bg-gray-100 text-gray-600 uppercase">
                    <tr>
                        <th className="text-left py-2 px-3 font-semibold">Region</th>
                        <th className="py-2 px-3 font-semibold text-right">Prob</th>
                        <th className="py-2 px-3 font-semibold text-left w-24">Confidence</th>
                    </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 bg-white">
                    {(Object.entries(result.probabilities) as [string, number][]).map(([region, prob]) => {
                        const isHigh = result.regions.includes(region);
                        return (
                            <tr key={region} className={isHigh ? 'bg-teal-50/20' : ''}>
                                <td className={`text-left font-medium py-2 px-3 ${isHigh ? 'text-teal-800' : 'text-gray-600'}`}>{region}</td>
                                <td className="py-2 px-3 text-right font-mono">{prob.toFixed(3)}</td>
                                <td className="py-2 px-3">
                                    <div className="w-full bg-gray-100 rounded-full h-2">
                                        <div 
                                            className={`h-2 rounded-full transition-all duration-500 ${isHigh ? 'bg-teal-500' : 'bg-gray-300'}`} 
                                            style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                        ></div>
                                    </div>
                                </td>
                            </tr>
                        );
                    })}
                    </tbody>
                </table>
            </div>

            {/* Metadata */}
            <div className="text-[10px] text-gray-400 border-t pt-2 space-y-1 font-mono">
                <div className="flex gap-2"><span className="font-bold">Src:</span> {result.mapping_source}</div>
                <div className="flex gap-2"><span className="font-bold">FP:</span> <span className="truncate">{result.mapping_fingerprint}</span></div>
                <div className="flex gap-2"><span className="font-bold">Head:</span> {result.localization_head_type}</div>
            </div>

            {/* XAI */}
            <XaiViewer xai={result.xai} baseUrl={baseUrl} />
          </div>
        )}
      </div>
    </div>
  );
}