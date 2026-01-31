import React, { useState } from "react";
import { apiRequest } from "../lib/api";
import { HealthResponse, ReadyResponse } from "../lib/types";

interface Props {
  baseUrl: string;
  onReadyChange: (isReady: boolean) => void;
}

export default function HealthReady({ baseUrl, onReadyChange }: Props) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readyData, setReadyData] = useState<ReadyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkSystem = async () => {
    setLoading(true);
    setError(null);
    try {
      const [hRes, rRes] = await Promise.all([
        apiRequest<HealthResponse>(baseUrl, "/health"),
        apiRequest<ReadyResponse>(baseUrl, "/ready"),
      ]);

      setHealth(hRes);
      setReadyData(rRes);
      onReadyChange(rRes.ready);
    } catch (err: any) {
      setError(err.message);
      onReadyChange(false);
      setHealth(null);
      setReadyData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white shadow rounded-lg p-4 mb-6 border-l-4 border-indigo-500">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-indigo-500"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          System Status
        </h2>
        <button
          onClick={checkSystem}
          disabled={loading}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded text-sm disabled:opacity-50 transition flex items-center gap-2"
        >
          {loading && <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
          {loading ? "Checking..." : "Check Connection"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 p-3 rounded text-sm mb-3 border border-red-200">
          <strong>Connection Failed:</strong> {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        {/* Health */}
        <div className="bg-gray-50 p-3 rounded border border-gray-100">
          <span className="font-semibold block text-gray-600 mb-2">Health Endpoint</span>
          {health ? (
            <div>
              <span className="px-2 py-1 rounded-full bg-green-100 text-green-800 text-xs font-bold uppercase tracking-wider">
                {health.status}
              </span>
              <div className="text-xs text-gray-500 mt-2 font-mono">{health.timestamp}</div>
            </div>
          ) : <span className="text-gray-400 italic">No data</span>}
        </div>

        {/* Ready */}
        <div className="bg-gray-50 p-3 rounded border border-gray-100">
          <span className="font-semibold block text-gray-600 mb-2">Model Readiness</span>
          {readyData ? (
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${readyData.ready ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'}`}>
                  {readyData.ready ? "Ready" : "Not Ready"}
                </span>
                <span className="text-gray-600 italic text-xs">"{readyData.message}"</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 text-xs">
                {Object.entries(readyData.models_loaded).map(([key, loaded]) => (
                  <div key={key} className={`text-center py-1 px-2 rounded border font-medium ${loaded ? 'bg-white border-green-200 text-green-700' : 'bg-white border-red-200 text-red-700'}`}>
                    {key}
                  </div>
                ))}
              </div>
            </div>
          ) : <span className="text-gray-400 italic">No data</span>}
        </div>
      </div>
    </div>
  );
}
