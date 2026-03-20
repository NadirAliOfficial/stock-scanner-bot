#!/usr/bin/env python3
"""
Stock Scanner — GUI wrapper
Connects to IBKR, runs the scan, and streams results to an activity log.
"""

import json
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ── Config persistence ─────────────────────────────────────────────────────────

CONFIG_FILE = Path(os.path.expanduser("~")) / ".stock_scanner_config.json"

DEFAULT_CONFIG = {
    "ib_host":      "127.0.0.1",
    "ib_port":      4001,
    "ib_client_id": 3,
    "watchlist":    "watchlist.csv",
    "output_dir":   "output",
}


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ── Main application ──────────────────────────────────────────────────────────

class ScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.running = False

        self.title("Stock Scanner Bot")
        self.minsize(720, 520)
        self.resizable(True, True)
        self._center_window(780, 600)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _center_window(self, w, h):
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        self.configure(bg="#1e1e2e")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".",             background="#1e1e2e", foreground="#e0e0e0",
                         font=("Segoe UI", 10))
        style.configure("TFrame",        background="#1e1e2e")
        style.configure("TLabel",        background="#1e1e2e", foreground="#e0e0e0")
        style.configure("TLabelframe",   background="#1e1e2e", foreground="#888",
                         bordercolor="#444")
        style.configure("TLabelframe.Label", background="#1e1e2e", foreground="#aaa",
                         font=("Segoe UI", 9, "bold"))
        style.configure("TEntry",        fieldbackground="#2a2a3e", foreground="#e0e0e0",
                         insertcolor="#e0e0e0", bordercolor="#555")
        style.configure("TButton",       background="#3c3f6e", foreground="#e0e0e0",
                         padding=(10, 5))
        style.map("TButton",
                  background=[("active", "#5a5faa"), ("disabled", "#2a2a3e")],
                  foreground=[("disabled", "#555")])
        style.configure("Run.TButton",   background="#2d7a3a", foreground="#ffffff",
                         font=("Segoe UI", 11, "bold"), padding=(16, 8))
        style.map("Run.TButton",
                  background=[("active", "#3da04e"), ("disabled", "#1a4a22")])
        style.configure("Stop.TButton",  background="#7a2d2d", foreground="#ffffff",
                         font=("Segoe UI", 11, "bold"), padding=(16, 8))
        style.map("Stop.TButton",
                  background=[("active", "#a03d3d")])

        # ── Top settings panel ────────────────────────────────────────────────
        top = ttk.Frame(self, padding=(12, 10, 12, 0))
        top.pack(fill="x")

        # Connection frame
        conn = ttk.LabelFrame(top, text="  IB Gateway / TWS Connection  ", padding=(12, 8))
        conn.pack(fill="x", pady=(0, 8))
        conn.columnconfigure(1, weight=1)
        conn.columnconfigure(3, weight=1)

        ttk.Label(conn, text="Host:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self._host_var = tk.StringVar(value=self.cfg["ib_host"])
        host_e = ttk.Entry(conn, textvariable=self._host_var, width=18)
        host_e.grid(row=0, column=1, sticky="ew", padx=(0, 20))

        ttk.Label(conn, text="Port:").grid(row=0, column=2, sticky="e", padx=(0, 6))
        self._port_var = tk.StringVar(value=str(self.cfg["ib_port"]))
        port_e = ttk.Entry(conn, textvariable=self._port_var, width=8)
        port_e.grid(row=0, column=3, sticky="w")

        ttk.Label(conn, text="Client ID:", font=("Segoe UI", 9)).grid(
            row=0, column=4, sticky="e", padx=(20, 6))
        self._cid_var = tk.StringVar(value=str(self.cfg["ib_client_id"]))
        ttk.Entry(conn, textvariable=self._cid_var, width=6).grid(row=0, column=5, sticky="w")

        # Port hint
        hint = ttk.Label(conn,
                         text="IB Gateway live: 4001  |  IB Gateway paper: 4002  |  TWS live: 7496  |  TWS paper: 7497",
                         foreground="#666", font=("Segoe UI", 8))
        hint.grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        # Files frame
        files = ttk.LabelFrame(top, text="  Files  ", padding=(12, 8))
        files.pack(fill="x", pady=(0, 8))
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Watchlist:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self._watchlist_var = tk.StringVar(value=self.cfg["watchlist"])
        ttk.Entry(files, textvariable=self._watchlist_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(files, text="Browse…", command=self._pick_watchlist).grid(row=0, column=2)

        ttk.Label(files, text="Output folder:").grid(row=1, column=0, sticky="e",
                                                      padx=(0, 6), pady=(6, 0))
        self._output_var = tk.StringVar(value=self.cfg["output_dir"])
        ttk.Entry(files, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 6), pady=(6, 0))
        ttk.Button(files, text="Browse…", command=self._pick_output).grid(
            row=1, column=2, pady=(6, 0))

        # ── Run / Stop button row ─────────────────────────────────────────────
        btn_row = ttk.Frame(self, padding=(12, 0))
        btn_row.pack(fill="x")
        self._run_btn = ttk.Button(btn_row, text="▶  Run Scan", style="Run.TButton",
                                   command=self._start_scan)
        self._run_btn.pack(side="left")
        self._stop_btn = ttk.Button(btn_row, text="■  Stop", style="Stop.TButton",
                                    command=self._request_stop, state="disabled")
        self._stop_btn.pack(side="left", padx=(10, 0))

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(btn_row, textvariable=self._status_var, foreground="#888",
                  font=("Segoe UI", 9)).pack(side="left", padx=(16, 0))

        ttk.Button(btn_row, text="Clear Log", command=self._clear_log).pack(side="right")
        ttk.Button(btn_row, text="Open Output Folder",
                   command=self._open_output).pack(side="right", padx=(0, 8))

        # ── Activity log ──────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="  Activity Log  ", padding=(8, 6))
        log_frame.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self._log = scrolledtext.ScrolledText(
            log_frame, state="disabled", wrap="word",
            bg="#12121e", fg="#e0e0e0", insertbackground="#e0e0e0",
            font=("Consolas", 9), relief="flat", borderwidth=0,
        )
        self._log.grid(row=0, column=0, sticky="nsew")
        self._log.tag_config("summary", foreground="#7ec8e3",
                             font=("Consolas", 9, "bold"))
        self._log.tag_config("progress", foreground="#a0d8a0")
        self._log.tag_config("err", foreground="#ff5555")
        self._log.tag_config("lvl_20", foreground="#e0e0e0")
        self._log.tag_config("lvl_30", foreground="#f0a500")
        self._log.tag_config("lvl_40", foreground="#ff5555")

        # ── Progress bar ──────────────────────────────────────────────────────
        prog_frame = ttk.Frame(self, padding=(12, 0, 12, 4))
        prog_frame.pack(fill="x")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate", length=100)
        self._progress.pack(fill="x")
        style.configure("TProgressbar", troughcolor="#2a2a3e", background="#3da04e",
                        thickness=6)

        self._log_info("Stock Scanner Bot ready.")
        self._log_info("Configure your connection settings above, then click  ▶ Run Scan.")

    # ── File pickers ──────────────────────────────────────────────────────────

    def _pick_watchlist(self):
        path = filedialog.askopenfilename(
            title="Select watchlist file",
            filetypes=[("Watchlist files", "*.csv *.txt *.json"), ("All files", "*.*")],
        )
        if path:
            self._watchlist_var.set(path)

    def _pick_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._output_var.set(path)

    def _open_output(self):
        folder = os.path.abspath(self._output_var.get() or "output")
        if not os.path.isdir(folder):
            messagebox.showinfo("Output Folder",
                                f"The folder does not exist yet:\n{folder}\n\n"
                                "It will be created when you run a scan.")
            return
        import subprocess
        if sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ── Logging helpers ───────────────────────────────────────────────────────

    def _log_write(self, msg, tag=None):
        def _do():
            self._log.config(state="normal")
            self._log.insert("end", msg + "\n", tag or "")
            self._log.see("end")
            self._log.config(state="disabled")
        self.after(0, _do)

    def _log_info(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_write(f"{ts}  INFO      {msg}")

    def _log_error(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_write(f"{ts}  ERROR     {msg}", "err")

    def _clear_log(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_inputs(self) -> dict | None:
        host = self._host_var.get().strip()
        if not host:
            messagebox.showerror("Invalid Input", "Please enter a host address.")
            return None

        try:
            port = int(self._port_var.get().strip())
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Port",
                                 "Port must be a number between 1 and 65535.\n\n"
                                 "Common values:\n"
                                 "  IB Gateway live  → 4001\n"
                                 "  IB Gateway paper → 4002\n"
                                 "  TWS live         → 7496\n"
                                 "  TWS paper        → 7497")
            return None

        try:
            client_id = int(self._cid_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Client ID", "Client ID must be a whole number.")
            return None

        watchlist = self._watchlist_var.get().strip()
        if not watchlist:
            messagebox.showerror("Missing Watchlist", "Please select a watchlist file.")
            return None
        if not os.path.isfile(watchlist):
            messagebox.showerror("File Not Found",
                                 f"Watchlist file not found:\n{watchlist}\n\n"
                                 "Please check the path or browse for the correct file.")
            return None

        output_dir = self._output_var.get().strip() or "output"

        return {
            "ib_host":      host,
            "ib_port":      port,
            "ib_client_id": client_id,
            "watchlist":    watchlist,
            "output_dir":   output_dir,
        }

    # ── Scan execution ────────────────────────────────────────────────────────

    def _start_scan(self):
        cfg = self._validate_inputs()
        if cfg is None:
            return
        save_config(cfg)

        self.running = True
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status_var.set("Connecting to IBKR…")

        self._scan_thread = threading.Thread(target=self._run_scan, args=(cfg,), daemon=True)
        self._scan_thread.start()

    def _request_stop(self):
        self.running = False
        self._status_var.set("Stopping…")
        self._log_write("─── Stop requested ───", "summary")
        if hasattr(self, "_proc") and self._proc.poll() is None:
            self._proc.terminate()

    def _run_scan(self, cfg: dict):
        """Run scanner.py as a subprocess and stream its output to the log."""
        import subprocess

        self._status_var_set("Starting scan…")
        os.makedirs(cfg["output_dir"], exist_ok=True)

        python = sys.executable
        cmd = [
            python, "-u", "scanner.py",
            "--host",      cfg["ib_host"],
            "--port",      str(cfg["ib_port"]),
            "--client-id", str(cfg["ib_client_id"]),
            "--watchlist", cfg["watchlist"],
            "--output",    cfg["output_dir"],
        ]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Launch Error", str(e)))
            self._scan_done(success=False)
            return

        # Stream output line by line into the log
        in_traceback = False
        error_lines = []
        for line in self._proc.stdout:
            if not self.running:
                self._proc.terminate()
                break
            line = line.rstrip()
            if not line:
                continue

            # Detect traceback blocks
            if line.startswith("Traceback (most recent call"):
                in_traceback = True
                error_lines = [line]
                self._log_write(line, "lvl_40")
                continue
            if in_traceback:
                error_lines.append(line)
                self._log_write(line, "lvl_40")
                # Traceback ends at the final exception line (not indented, not "The above...")
                if not line.startswith(" ") and not line.startswith("The above"):
                    in_traceback = False
                continue

            # Colour-code by level keyword
            if "ERROR" in line or "error" in line.lower() or "failed" in line.lower():
                tag = "lvl_40"
                error_lines.append(line)
            elif "WARNING" in line or "warning" in line.lower():
                tag = "lvl_30"
            elif "CANDIDATE" in line or "★" in line:
                tag = "summary"
            elif "Processing:" in line or "▶" in line:
                tag = "progress"
            else:
                tag = "lvl_20"

            # Parse progress from "Processing: AAPL" lines
            if "Processing:" in line:
                sym = line.split("Processing:")[-1].strip()
                self._status_var_set(f"Scanning {sym}…")

            self._log_write(line, tag)

        self._proc.wait()
        rc = self._proc.returncode

        if rc == 0:
            summary = (
                "────────────────────────────────────────────────────────\n"
                "  Scan complete. Check output folder for CSV and Excel.\n"
                "────────────────────────────────────────────────────────"
            )
            self._scan_done(success=True, summary=summary)
        else:
            # Show a user-friendly popup for connection / crash errors
            if error_lines:
                # Extract the final exception type (last non-indented line)
                exc_line = error_lines[-1].strip()
                if "TimeoutError" in exc_line or "connection" in exc_line.lower():
                    msg = (
                        f"IBKR API handshake timed out.\n\n"
                        f"The socket on {cfg['ib_host']}:{cfg['ib_port']} is open "
                        f"but IBKR did not complete the API handshake.\n\n"
                        f"Please check:\n"
                        f"  • API connections are enabled in IBKR\n"
                        f"    (Configure → API → Settings)\n"
                        f"  • 'Allow connections from localhost' is checked\n"
                        f"  • Client ID {cfg['ib_client_id']} is not already in use\n"
                        f"  • You are not behind a firewall blocking the connection"
                    )
                else:
                    msg = f"Scanner exited with an error:\n\n{exc_line}"
                self.after(0, lambda m=msg: messagebox.showerror("Scan Error", m))
            self._scan_done(success=False)

    def _status_var_set(self, msg):
        self.after(0, lambda: self._status_var.set(msg))

    def _scan_done(self, success: bool, summary: str | None = None):
        self.running = False

        def _update():
            self._run_btn.config(state="normal")
            self._stop_btn.config(state="disabled")
            self._progress.config(value=0)
            if success and summary:
                self._log.config(state="normal")
                self._log.insert("end", summary + "\n", "summary")
                self._log.see("end")
                self._log.config(state="disabled")
                self._status_var.set("Scan complete.")
            elif success:
                self._status_var.set("Scan finished (no output).")
            else:
                self._status_var.set("Scan failed — see log for details.")

        self.after(0, _update)

    # ── Window close ──────────────────────────────────────────────────────────

    def _on_close(self):
        if self.running:
            if not messagebox.askyesno("Scan Running",
                                       "A scan is currently running.\n\n"
                                       "Close anyway?"):
                return
        try:
            port = int(self._port_var.get())
        except (ValueError, TypeError):
            port = 4001
        try:
            client_id = int(self._cid_var.get())
        except (ValueError, TypeError):
            client_id = 3
        save_config({
            "ib_host":      self._host_var.get(),
            "ib_port":      port,
            "ib_client_id": client_id,
            "watchlist":    self._watchlist_var.get(),
            "output_dir":   self._output_var.get(),
        })
        self.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ScannerApp()
    app.mainloop()
