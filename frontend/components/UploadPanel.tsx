import React, { useState, ChangeEvent } from "react";
import { apiRequest } from "../lib/api";
import { SuperclassResponse, LocalizationResponse } from "../lib/types";

interface UploadPanelProps {
  baseUrl: string;
  disabled: boolean;
  onSuperclassResult: (result: SuperclassResponse) => void;
  onLocalizationResult: (result: LocalizationResponse) => void;
  onError: (error: string) => void;
  onLoadingChange: (loading: boolean) => void;
}

export default function UploadPanel({
  baseUrl,
  disabled,
  onSuperclassResult,
  onLocalizationResult,
  onError,
  onLoadingChange,
}: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [ensembleWeight, setEnsembleWeight] = useState(0.5);
  const [explain, setExplain] = useState(false);
  const [sanityCheck, setSanityCheck] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setFileError(null);
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (selectedFile.size > 10 * 1024 * 1024) {
        setFileError("Dosya boyutu 10MB limitini aşıyor.");
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
    onLoadingChange(true);
    onError("");
    setFileError(null);

    const formData = new FormData();
    formData.append("file", file);

    const superclassParams = new URLSearchParams({
      ensemble_weight: ensembleWeight.toString(),
      explain: explain.toString(),
      sanity_check: (explain && sanityCheck).toString(),
    });

    const localizationParams = new URLSearchParams({
      threshold: "0.5",
      explain: explain.toString(),
    });

    try {
      const [superRes, locRes] = await Promise.all([
        apiRequest<SuperclassResponse>(
          baseUrl,
          `/predict/superclass?${superclassParams.toString()}`,
          { method: "POST", body: formData },
          60000
        ),
        apiRequest<LocalizationResponse>(
          baseUrl,
          `/predict/mi-localization?${localizationParams.toString()}`,
          { method: "POST", body: (() => { const fd = new FormData(); fd.append("file", file); return fd; })() },
          60000
        ),
      ]);
      onSuperclassResult(superRes);
      onLocalizationResult(locRes);
    } catch (err: any) {
      onError(err?.message || String(err) || 'Bilinmeyen hata');
    } finally {
      setLoading(false);
      onLoadingChange(false);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm">
      <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-700">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100">
          EKG Analizi
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Dosya yükleyerek tahmin başlatın
        </p>
      </div>

      <div className="p-4 space-y-4">
        <div>
          <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-1">
            EKG Dosyası Yükle
          </label>
          <input
            type="file"
            accept=".npy,.npz"
            onChange={handleFileChange}
            disabled={disabled}
            className="block w-full text-sm text-slate-500 dark:text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600 disabled:opacity-50 cursor-pointer"
          />
          <p className="text-[10px] text-slate-400 dark:text-slate-500 mt-1 text-right">
            Maks: 10MB (.npy/.npz)
          </p>
          {fileError && (
            <p className="text-xs text-red-500 mt-1">{fileError}</p>
          )}
        </div>

        <div>
          <div className="flex justify-between items-center mb-1">
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300">
              Ensemble Ağırlığı
            </label>
            <span className="text-xs font-mono bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 px-2 py-0.5 rounded border border-slate-200 dark:border-slate-600">
              {ensembleWeight}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={ensembleWeight}
            onChange={(e) => setEnsembleWeight(parseFloat(e.target.value))}
            disabled={disabled}
            className="w-full accent-blue-500 h-2 bg-slate-200 dark:bg-slate-600 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-[10px] text-slate-400 dark:text-slate-500 px-1">
            <span>CNN</span>
            <span>Ensemble</span>
            <span>XGB</span>
          </div>
        </div>

        <div className="flex gap-6 border-t border-slate-200 dark:border-slate-700 pt-3">
          <label className="flex items-center space-x-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
              disabled={disabled}
              className="rounded text-blue-500 focus:ring-blue-500"
            />
            <span className="font-medium">XAI Açıklama</span>
          </label>
          <label className="flex items-center space-x-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={sanityCheck}
              onChange={(e) => setSanityCheck(e.target.checked)}
              disabled={disabled || !explain}
              className="rounded text-blue-500 focus:ring-blue-500 disabled:text-slate-400"
            />
            <span className={!explain ? "text-slate-400 dark:text-slate-600" : "font-medium"}>
              Kalite Kontrolü
            </span>
          </label>
        </div>

        <button
          onClick={handleSubmit}
          disabled={!file || disabled || loading}
          className="w-full bg-blue-500 hover:bg-blue-600 text-white py-2.5 rounded shadow-sm font-semibold disabled:bg-slate-300 dark:disabled:bg-slate-600 disabled:shadow-none disabled:cursor-not-allowed transition flex justify-center items-center gap-2"
        >
          {loading && (
            <svg
              className="animate-spin h-4 w-4 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          )}
          {loading ? "Analiz Ediliyor..." : "Tahmin Yap"}
        </button>
      </div>
    </div>
  );
}
