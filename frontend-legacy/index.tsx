import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import ThemeProvider from "./components/ThemeProvider";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import SystemStatus from "./components/SystemStatus";
import PredictionResult from "./components/PredictionResult";
import ConsistencyPanel from "./components/ConsistencyPanel";
import LocalizationPanel from "./components/LocalizationPanel";
import XaiViewer from "./components/XaiViewer";
import { SuperclassResponse, LocalizationResponse } from "./lib/types";

function App() {
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000");
  const [systemReady, setSystemReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [superclassResult, setSuperclassResult] = useState<SuperclassResponse | null>(null);
  const [localizationResult, setLocalizationResult] = useState<LocalizationResponse | null>(null);

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-900 dark:text-slate-100 font-sans transition-colors duration-200">
        <Header
          baseUrl={baseUrl}
          onBaseUrlChange={setBaseUrl}
          systemReady={systemReady}
        />

        <main className="max-w-7xl mx-auto px-4 py-6">
          {!systemReady && (
            <div className="mb-6 bg-amber-500/10 border border-amber-500/20 p-4 text-amber-600 dark:text-amber-400 text-sm rounded-lg flex items-start gap-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mt-0.5 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <div>
                <h3 className="font-bold">Sistem Kullanılamıyor</h3>
                <p className="mt-1">
                  Tahmin özellikleri devre dışı. Lütfen API adresini kontrol edin ve sunucunun çalıştığından emin olun.
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="mb-6 bg-red-500/10 border border-red-500/20 p-4 text-red-500 dark:text-red-400 text-sm rounded-lg flex items-start gap-2">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 flex-shrink-0">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span className="break-all">{error}</span>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Panel */}
            <div className="space-y-4">
              <UploadPanel
                baseUrl={baseUrl}
                disabled={!systemReady || loading}
                onSuperclassResult={setSuperclassResult}
                onLocalizationResult={setLocalizationResult}
                onError={setError}
                onLoadingChange={setLoading}
              />
              <SystemStatus baseUrl={baseUrl} onReadyChange={setSystemReady} />
            </div>

            {/* Right Panel */}
            <div className="lg:col-span-2 space-y-4">
              {superclassResult && (
                <PredictionResult result={superclassResult} />
              )}

              {superclassResult?.consistency && (
                <ConsistencyPanel consistency={superclassResult.consistency} />
              )}

              {localizationResult && (
                <LocalizationPanel result={localizationResult} />
              )}

              {superclassResult?.xai && superclassResult.xai.enabled && (
                <XaiViewer xai={superclassResult.xai} baseUrl={baseUrl} />
              )}

              {localizationResult?.xai && localizationResult.xai.enabled && (
                <XaiViewer xai={localizationResult.xai} baseUrl={baseUrl} />
              )}

              {!superclassResult && !localizationResult && (
                <div className="flex items-center justify-center h-64 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 border-dashed">
                  <div className="text-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mx-auto text-slate-300 dark:text-slate-600 mb-3">
                      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                    <p className="text-slate-400 dark:text-slate-500 text-sm">
                      Sonuçları görmek için bir EKG dosyası yükleyin
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>

        <footer className="text-center text-slate-400 dark:text-slate-600 text-xs py-8 mt-8 border-t border-slate-200 dark:border-slate-800">
          <p>CardioGuard-AI Klinik Karar Destek Sistemi</p>
        </footer>
      </div>
    </ThemeProvider>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
