#!/usr/bin/env python3
"""
GUI Application for Error Detection Model

A graphical user interface for analyzing code files and detecting potential errors
using static analysis and machine learning models.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from error_predictor import ErrorDetectionModel, ErrorType
from error_reporter import ErrorReporter


class ErrorDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Error Detection Model - GUI")
        self.root.geometry("800x600")
        self.root.minsize(600, 400)

        # Initialize model and reporter
        self.model = None
        self.reporter = ErrorReporter()

        # Setup GUI components
        self.setup_ui()

        # Initialize model (in background)
        self.initialize_model()

    def setup_ui(self):
        """Setup the user interface components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Selected File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))

        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly")
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        self.browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, sticky=tk.W)

        # Analysis options section
        options_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Label(options_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        confidence_scale = ttk.Scale(options_frame, from_=0.0, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))

        self.confidence_label = ttk.Label(options_frame, text="0.5")
        self.confidence_label.grid(row=0, column=2, sticky=tk.W)
        confidence_scale.configure(command=self.update_confidence_label)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        self.analyze_button = ttk.Button(button_frame, text="Analyze File", command=self.analyze_file, state="disabled")
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 5))

        self.clear_button = ttk.Button(button_frame, text="Clear Results", command=self.clear_results)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 5))

        self.save_button = ttk.Button(button_frame, text="Save Report", command=self.save_report, state="disabled")
        self.save_button.pack(side=tk.LEFT)

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, state="disabled")
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Store results for saving
        self.current_results = []

    def update_confidence_label(self, value):
        """Update the confidence threshold label"""
        self.confidence_label.config(text=f"{float(value):.2f}")

    def initialize_model(self):
        """Initialize the error detection model in a background thread"""
        def init_model():
            try:
                self.status_var.set("Initializing model...")
                self.model = ErrorDetectionModel()
                self.status_var.set("Model initialized successfully")
                self.analyze_button.config(state="normal")
            except Exception as e:
                self.status_var.set(f"Error initializing model: {str(e)}")
                messagebox.showerror("Model Error", f"Failed to initialize model:\n{str(e)}")

        threading.Thread(target=init_model, daemon=True).start()

    def browse_file(self):
        """Open file dialog to select a code file"""
        filetypes = [
            ("Python files", "*.py"),
            ("JavaScript files", "*.js *.jsx"),
            ("TypeScript files", "*.ts *.tsx"),
            ("C/C++ files", "*.c *.cpp *.cc *.cxx *.c++ *.h *.hpp"),
            ("Java files", "*.java"),
            ("Go files", "*.go"),
            ("Rust files", "*.rs"),
            ("PHP files", "*.php"),
            ("Ruby files", "*.rb"),
            ("Swift files", "*.swift"),
            ("Kotlin files", "*.kt"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select a code file to analyze",
            filetypes=filetypes
        )

        if filename:
            self.file_path_var.set(filename)
            self.status_var.set(f"Selected: {Path(filename).name}")

    def analyze_file(self):
        """Analyze the selected file for potential errors"""
        file_path = self.file_path_var.get()

        if not file_path:
            messagebox.showwarning("No File Selected", "Please select a file to analyze.")
            return

        if not Path(file_path).exists():
            messagebox.showerror("File Not Found", f"The selected file does not exist:\n{file_path}")
            return

        if not self.model:
            messagebox.showerror("Model Not Ready", "The error detection model is not initialized yet.")
            return

        # Disable button during analysis
        self.analyze_button.config(state="disabled")
        self.status_var.set("Analyzing file...")

        def perform_analysis():
            try:
                # Analyze the file
                result = self.model.predict_file(file_path)
                self.current_results = [result]

                # Filter by confidence threshold
                confidence_threshold = self.confidence_var.get()

                # Check if we should filter errors based on confidence
                # If result has multiple errors, filter them individually
                if result.all_errors and len(result.all_errors) > 1:
                    # Filter individual errors by confidence
                    filtered_errors = [
                        err for err in result.all_errors
                        if err.get('confidence', result.confidence) >= confidence_threshold
                    ]

                    if filtered_errors:
                        # Update result with filtered errors
                        from error_predictor import PredictionResult
                        filtered_result = PredictionResult(
                            file_path=result.file_path,
                            language=result.language,
                            error_type=result.error_type,
                            confidence=result.confidence,
                            line_number=result.line_number,
                            error_message=result.error_message,
                            suggestions=result.suggestions,
                            all_errors=filtered_errors
                        )
                        filtered_results = [filtered_result]
                    else:
                        filtered_results = []
                else:
                    # Single error - use original filtering logic
                    if result.confidence < confidence_threshold:
                        filtered_results = []
                    else:
                        filtered_results = [result]

                # Format results for display
                if filtered_results:
                    report = self.reporter.format_console_report(filtered_results)
                else:
                    report = f"No issues found above confidence threshold ({confidence_threshold:.2f})\n\n"
                    report += f"Analysis completed for: {Path(file_path).name}\n"
                    if result.all_errors:
                        report += f"Total errors detected: {len(result.all_errors)} (all below threshold)"
                    else:
                        report += f"Detected error type: {result.error_type.value}\n"
                        report += f"Confidence: {result.confidence:.3f} (below threshold)"

                # Update UI in main thread
                self.root.after(0, self.display_results, report)

            except Exception as e:
                error_msg = f"Error analyzing file: {str(e)}"
                self.root.after(0, self.display_error, error_msg)

        # Run analysis in background thread
        threading.Thread(target=perform_analysis, daemon=True).start()

    def display_results(self, report):
        """Display analysis results in the text widget"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, report)
        self.results_text.config(state="disabled")

        # Re-enable button and update status
        self.analyze_button.config(state="normal")
        self.save_button.config(state="normal")
        self.status_var.set("Analysis completed")

    def display_error(self, error_msg):
        """Display error message"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, error_msg)
        self.results_text.config(state="disabled")

        # Re-enable button and update status
        self.analyze_button.config(state="normal")
        self.status_var.set("Analysis failed")
        messagebox.showerror("Analysis Error", error_msg)

    def clear_results(self):
        """Clear the results display"""
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state="disabled")

        self.current_results = []
        self.save_button.config(state="disabled")
        self.status_var.set("Results cleared")

    def save_report(self):
        """Save the analysis results to a file"""
        if not self.current_results:
            messagebox.showwarning("No Results", "No analysis results to save.")
            return

        filetypes = [
            ("JSON files", "*.json"),
            ("CSV files", "*.csv"),
            ("HTML files", "*.html"),
            ("Text files", "*.txt"),
        ]

        filename = filedialog.asksaveasfilename(
            title="Save analysis report",
            filetypes=filetypes,
            defaultextension=".json"
        )

        if filename:
            try:
                file_ext = Path(filename).suffix.lower()

                if file_ext == '.json':
                    self.reporter.save_json_report(self.current_results, filename)
                elif file_ext == '.csv':
                    self.reporter.save_csv_report(self.current_results, filename)
                elif file_ext == '.html':
                    self.reporter.generate_html_report(self.current_results, filename)
                else:  # .txt or other
                    report = self.reporter.format_console_report(self.current_results)
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)

                self.status_var.set(f"Report saved to {Path(filename).name}")
                messagebox.showinfo("Save Successful", f"Report saved to:\n{filename}")

            except Exception as e:
                error_msg = f"Error saving report: {str(e)}"
                self.status_var.set("Save failed")
                messagebox.showerror("Save Error", error_msg)


def main():
    """Main function to run the GUI application"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run the GUI
    root = tk.Tk()
    app = ErrorDetectionGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()