import { type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { Moon, Sun, Settings, Activity, Info } from "lucide-react";
import { useTheme } from "@/hooks/useTheme";
import { Button } from "@/components/ui/button";
import logo from "@/assets/cardioguard-logo.png";
import { DISCLAIMER } from "@/lib/glossary";

export function AppShell({ children }: { children: ReactNode }) {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-card/80 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-[1500px] items-center justify-between px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2.5">
            <img src={logo} alt="CardioGuard-AI logosu" width={36} height={36} className="h-9 w-9" />
            <div className="leading-tight">
              <span className="block font-display text-base font-semibold tracking-tight text-foreground">
                CardioGuard<span className="text-primary">-AI</span>
              </span>
              <span className="hidden text-[11px] text-muted-foreground sm:block">
                Açıklanabilir EKG Analiz Platformu
              </span>
            </div>
          </Link>

          <nav className="flex items-center gap-1">
            <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground">
              <Link to="/" activeOptions={{ exact: true }} activeProps={{ className: "text-foreground" }}>
                <Activity className="h-4 w-4" />
                <span className="hidden sm:inline">Analiz</span>
              </Link>
            </Button>
            <Button asChild variant="ghost" size="sm" className="gap-1.5 text-muted-foreground">
              <Link to="/about" activeProps={{ className: "text-foreground" }}>
                <Info className="h-4 w-4" />
                <span className="hidden sm:inline">Hakkında</span>
              </Link>
            </Button>
            <Button asChild variant="ghost" size="icon" aria-label="Ayarlar">
              <Link to="/settings" activeProps={{ className: "text-foreground" }}>
                <Settings className="h-4 w-4" />
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Açık temaya geç" : "Koyu temaya geç"}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          </nav>
        </div>
      </header>

      <main className="flex-1">{children}</main>

      <footer className="border-t border-border bg-card/50 py-4">
        <div className="mx-auto w-full max-w-[1500px] px-4 text-center text-xs text-muted-foreground sm:px-6">
          ⚠️ {DISCLAIMER} Nihai değerlendirme hekim tarafından yapılmalıdır.
        </div>
      </footer>
    </div>
  );
}
