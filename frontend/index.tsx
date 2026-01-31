import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import HealthReady from "./components/HealthReady";
import SuperclassPanel from "./components/SuperclassPanel";
import LocalizationPanel from "./components/LocalizationPanel";

function App() {
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000");
  const [systemReady, setSystemReady] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 pb-20">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-20 border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-3">
             <div className="h-9 w-9 bg-gradient-to-br from-red-600 to-red-700 rounded-lg shadow-sm flex items-center justify-center text-white font-bold text-lg">C</div>
             <div>
                <h1 className="text-xl font-bold text-slate-800 leading-tight">CardioGuard<span className="text-red-600">-AI</span></h1>
                <p className="text-[10px] text-gray-500 font-medium tracking-wide uppercase">Advanced ECG Diagnostics</p>
             </div>
          </div>
          
          <div className="flex items-center gap-3 w-full md:w-auto bg-gray-50 p-1.5 rounded-md border border-gray-200">
            <label className="text-xs font-semibold text-gray-500 whitespace-nowrap px-2">API HOST</label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="bg-white border border-gray-300 rounded px-3 py-1 text-sm text-gray-700 w-full md:w-64 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition font-mono"
            />
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        
        {/* Status Check */}
        <HealthReady baseUrl={baseUrl} onReadyChange={setSystemReady} />

        {/* Not Ready Warning */}
        {!systemReady && (
            <div className="mb-8 bg-amber-50 border-l-4 border-amber-400 p-4 text-amber-800 text-sm shadow-sm rounded-r-md flex items-start gap-3">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-amber-500 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <div>
                    <h3 className="font-bold">System Unavailable</h3>
                    <p className="mt-1">Prediction features are disabled because the backend is not ready or reachable. Please verify the API URL above and ensure the server is running.</p>
                </div>
            </div>
        )}

        {/* Prediction Grid */}
        <div className={`grid grid-cols-1 lg:grid-cols-2 gap-8 transition-opacity duration-300 ${!systemReady ? 'opacity-50 pointer-events-none grayscale-[0.5]' : ''}`}>
          <SuperclassPanel baseUrl={baseUrl} disabled={!systemReady} />
          <LocalizationPanel baseUrl={baseUrl} disabled={!systemReady} />
        </div>
      </main>

      {/* Footer */}
      <footer className="text-center text-gray-400 text-xs py-8 mt-8 border-t border-gray-200">
        <p>&copy; 2025 CardioGuard-AI. Clinical Decision Support System Demo.</p>
      </footer>
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
