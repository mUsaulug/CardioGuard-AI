# Phase 5: Frontend Integration — Kapsamlı Analiz

**Generated Date:** 2026-01-31
**Kaynak:** `frontend/` klasörü
**Stack:** React 19.2.4, Vite 6.2.0, TypeScript 5.8.2

---

## 1. Frontend Mimarisi

### 1.1 Teknoloji Stack

| Teknoloji | Versiyon | Rol |
| :--- | :--- | :--- |
| **React** | 19.2.4 | UI Library (en güncel major) |
| **Vite** | 6.2.0 | Build tool & dev server |
| **TypeScript** | 5.8.2 | Type-safe JavaScript |
| **CSS** | Vanilla | Styling (TailwindCSS yok) |

### 1.2 Proje Yapısı

```
frontend/
├── index.html              # HTML template
├── index.tsx               # React entry point (3791 bytes)
├── package.json            # Dependencies
├── package-lock.json       # Lock file (60KB)
├── vite.config.ts          # Bundler config
├── tsconfig.json           # TS config
├── .env.local              # Environment variables
├── components/             # React components (4 files)
│   ├── ECGUploader.tsx
│   ├── ResultDisplay.tsx
│   ├── XAIViewer.tsx
│   └── Header.tsx
└── lib/                    # Shared utilities
    ├── api.ts              # HTTP client (1455 bytes)
    └── types.ts            # Type definitions (1810 bytes)
```

---

## 2. Type Definitions — Backend Kontrat Uyumu

### 2.1 Tam Kaynak: `lib/types.ts`

```typescript
// === Health & Readiness ===

export interface HealthResponse {
  status: string;
  timestamp: string;
}

export interface ReadyResponse {
  ready: boolean;
  models_loaded: {
    superclass: boolean;
    localization: boolean;
    xgb: boolean;
    thresholds: boolean;
  };
  message: string;
}

// === XAI Artifacts ===

export interface Artifact {
  type: string;      // "report_png", "narrative_md"
  name: string;      // filename
  url: string;       // relative URL: /runs/{id}/path
  mime: string;      // MIME type
}

export interface XaiSchema {
  enabled: boolean;
  run_id: string | null;
  run_dir: string | null;
  artifacts: Artifact[];
  highlights: object[] | null;
  sanity: object | null;
}

// === Version Info ===

export interface Versions {
  model_hash: string;
  threshold_hash: string;
  api_version: string;
  timestamp: string;
}

// === Superclass Prediction ===

export interface SuperclassProbabilities {
  MI: number;
  STTC: number;
  CD: number;
  HYP: number;
  NORM: number;
}

export interface SuperclassResponse {
  mode: string;                          // "multilabel-superclass"
  probabilities: SuperclassProbabilities;
  predicted_labels: string[];            // ["MI", "STTC"]
  thresholds: {
    MI: number;
    STTC: number;
    CD: number;
    HYP: number;
  };
  primary: {
    label: string;
    confidence: number;
    rule: string;
  };
  sources: {
    cnn: SuperclassProbabilities;
    xgb: SuperclassProbabilities | null;
    ensemble: SuperclassProbabilities;
  };
  versions: Versions;
  xai: XaiSchema | null;
}

// === MI Localization ===

export interface LocalizationProbabilities {
  AMI: number;
  ASMI: number;
  ALMI: number;
  IMI: number;
  LMI: number;
}

export interface LocalizationResponse {
  mi_detected: boolean;
  regions: string[];                     // ["IMI", "ALMI"]
  probabilities: LocalizationProbabilities;
  label_space: string;
  labels: string[];
  mapping_source: string;
  mapping_fingerprint: string;
  localization_head_type: string;
  xai: XaiSchema | null;
}

// === Error Handling ===

export interface ApiError {
  error: string;
  detail?: string;
}
```

### 2.2 Kontrat Uyumu Tablosu

| Backend (Pydantic) | Frontend (TS) | Field Count | Uyum |
| :--- | :--- | :---: | :---: |
| `SuperclassPredictionResponse` | `SuperclassResponse` | 8 | ✅ 100% |
| `MILocalizationResponse` | `LocalizationResponse` | 9 | ✅ 100% |
| `PredictionProbabilities` | `SuperclassProbabilities` | 5 | ✅ 100% |
| `LocalizationProbabilities` | `LocalizationProbabilities` | 5 | ✅ 100% |
| `XAIInfo` | `XaiSchema` | 5 | ✅ 100% |
| `XAIArtifact` | `Artifact` | 4 | ✅ 100% |
| `VersionInfo` | `Versions` | 4 | ✅ 100% |

**Sonuç:** Frontend ve Backend kontratları **tam uyumlu**. Bu, ya OpenAPI codegen kullanıldığını ya da çok disiplinli manuel senkronizasyon yapıldığını gösterir.

---

## 3. API Client

### 3.1 Tam Kaynak: `lib/api.ts`

```typescript
/**
 * URL Temizleme
 * Trailing/leading slash'ları düzeltir
 */
export const cleanUrl = (base: string, path: string) => {
  const cleanBase = base.replace(/\/+$/, '');
  const cleanPath = path.replace(/^\\/+/, '');
  return `${cleanBase}/${cleanPath}`;
};

/**
 * Generic API Request
 * Type-safe, timeout destekli
 */
export async function apiRequest<T>(
  baseUrl: string,
  endpoint: string,
  options: RequestInit = {},
  timeoutMs: number = 30000
): Promise<T> {
  const url = cleanUrl(baseUrl, endpoint);
  
  // Timeout için AbortController
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(id);
    
    // Error handling
    if (!res.ok) {
      const errorText = await res.text();
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

/**
 * Text Artifact Fetch
 * Markdown narrative gibi text dosyalarını çeker
 */
export const fetchTextArtifact = async (
  baseUrl: string,
  artifactUrl: string
): Promise<string> => {
  const url = cleanUrl(baseUrl, artifactUrl);
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch artifact text");
  return res.text();
};
```

### 3.2 API Client Özellikleri

| Özellik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Timeout** | 30 saniye | AbortController ile |
| **Error Parsing** | JSON → text fallback | FastAPI detail field'ını yakalar |
| **Type Safety** | Generic `<T>` | Response type inference |
| **URL Handling** | cleanUrl() | Slash normalization |

---

## 4. Kullanım Örneği

### 4.1 Superclass Prediction

```typescript
import { apiRequest, SuperclassResponse } from './lib';

async function predictECG(file: File): Promise<SuperclassResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  return apiRequest<SuperclassResponse>(
    'http://localhost:8000',
    '/predict/superclass?explain=true',
    {
      method: 'POST',
      body: formData
    }
  );
}

// Kullanım
const result = await predictECG(selectedFile);
console.log(result.primary.label);        // "MI"
console.log(result.probabilities.MI);     // 0.8523
console.log(result.xai?.artifacts[0].url); // "/runs/.../report.png"
```

### 4.2 Artifact Görüntüleme

```typescript
// XAI Report PNG gösterme
const reportUrl = result.xai?.artifacts
  .find(a => a.type === 'report_png')?.url;

if (reportUrl) {
  const fullUrl = cleanUrl('http://localhost:8000', reportUrl);
  // <img src={fullUrl} />
}

// Narrative Markdown okuma
const narrativeUrl = result.xai?.artifacts
  .find(a => a.type === 'narrative_md')?.url;

if (narrativeUrl) {
  const markdown = await fetchTextArtifact('http://localhost:8000', narrativeUrl);
  // Markdown render
}
```

---

## 5. Build & Development

### 5.1 Komutlar

```bash
# Development server (HMR destekli)
cd frontend
npm install
npm run dev
# → http://localhost:5173

# Production build
npm run build
# → frontend/dist/ (static files)

# Preview production build
npm run preview
```

### 5.2 Vite Konfigürasyonu

**Kaynak:** `frontend/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
});
```

### 5.3 TypeScript Konfigürasyonu

**Kaynak:** `frontend/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "lib": ["DOM", "DOM.Iterable", "ESNext"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx",
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["*.tsx", "*.ts", "components/**/*", "lib/**/*"]
}
```

---

## 6. Environment Konfigürasyonu

**Kaynak:** `frontend/.env.local`

```bash
VITE_API_BASE_URL=http://localhost:8000
```

**Kullanım:**
```typescript
const API_BASE = import.meta.env.VITE_API_BASE_URL;
```

---

## 7. Bileşen Yapısı (İnferred)

```
┌─────────────────────────────────────────────────┐
│                    Header                       │
│    CardioGuard-AI | Health: ✓ | Ready: ✓       │
├─────────────────────────────────────────────────┤
│                                                 │
│    ┌───────────────────────────────────────┐    │
│    │          ECGUploader                  │    │
│    │   [Drag & Drop ECG File]              │    │
│    │   [Select .npz or .npy]               │    │
│    │   [☑ Generate XAI Explanation]        │    │
│    │   [    ANALYZE    ]                   │    │
│    └───────────────────────────────────────┘    │
│                                                 │
│    ┌───────────────────────────────────────┐    │
│    │          ResultDisplay                │    │
│    │   Primary: MI (85.2%)                 │    │
│    │   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓ MI: 85%             │    │
│    │   ▓▓▓▓          STTC: 23%             │    │
│    │   ▓▓            CD: 12%               │    │
│    │   ▓             HYP: 9%               │    │
│    └───────────────────────────────────────┘    │
│                                                 │
│    ┌───────────────────────────────────────┐    │
│    │            XAIViewer                  │    │
│    │   [12-Lead ECG with Heatmap]          │    │
│    │   [Top SHAP Features]                 │    │
│    │   [Narrative Summary]                 │    │
│    └───────────────────────────────────────┘    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 8. Özet: Frontend Güçlü ve Zayıf Yönler

| Kategori | Güçlü Yön | Eksik / Not |
| :--- | :--- | :--- |
| **Type Safety** | %100 Backend uyumu | - |
| **Modern Stack** | React 19, Vite 6, TS 5.8 | Cutting edge |
| **Error Handling** | Timeout + JSON parsing | - |
| **Bundle Size** | Minimal (no Tailwind) | Performant |
| **Testing** | - | Frontend testleri görünmüyor |
| **State Management** | - | Büyük uygulamalar için Redux/Zustand gerekebilir |
