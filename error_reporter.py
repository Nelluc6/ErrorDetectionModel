"""
Error Reporter Module

This module handles the formatting and visualization of error detection results.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import asdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ErrorReporter:
    """
    Handles reporting and visualization of error detection results
    """

    def __init__(self):
        self.results_history = []

    def format_console_report(self, results: List[Any]) -> str:
        """
        Format results for console output

        Args:
            results: List of PredictionResult objects

        Returns:
            Formatted string for console display
        """
        if not results:
            return "No files analyzed."

        report = []
        report.append("=" * 60)
        report.append("ERROR DETECTION REPORT")
        report.append("=" * 60)

        # Summary statistics
        total_files = len(results)
        error_files = sum(1 for r in results if r.error_type.value != 'no_error')
        error_rate = (error_files / total_files) * 100 if total_files > 0 else 0

        report.append(f"\nSUMMARY:")
        report.append(f"  Total files analyzed: {total_files}")
        report.append(f"  Files with potential errors: {error_files}")
        report.append(f"  Error rate: {error_rate:.1f}%")

        # Group by error type
        error_types = {}
        for result in results:
            error_type = result.error_type.value
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(result)

        report.append(f"\nERROR BREAKDOWN:")
        for error_type, files in error_types.items():
            report.append(f"  {error_type}: {len(files)} files")

        # Detailed results
        report.append(f"\nDETAILED RESULTS:")
        report.append("-" * 60)

        for result in results:
            report.append(f"\nFile: {result.file_path}")
            report.append(f"Language: {result.language}")

            # If we have multiple errors, show them all
            if result.all_errors and len(result.all_errors) > 1:
                report.append(f"\nFound {len(result.all_errors)} potential issues:")
                report.append("")

                for i, error in enumerate(result.all_errors, 1):
                    report.append(f"  Issue {i}:")
                    error_type = error.get('type', 'unknown')
                    if hasattr(error_type, 'value'):
                        error_type = error_type.value
                    report.append(f"    Type: {error_type}")
                    report.append(f"    Confidence: {error.get('confidence', result.confidence):.2f}")

                    if error.get('line'):
                        report.append(f"    Line: {error['line']}")

                    if error.get('message'):
                        report.append(f"    Message: {error['message']}")

                    if error.get('suggestions'):
                        report.append(f"    Suggestions:")
                        for suggestion in error['suggestions']:
                            report.append(f"      • {suggestion}")
                    report.append("")
            else:
                # Single error display
                report.append(f"Error Type: {result.error_type.value}")
                report.append(f"Confidence: {result.confidence:.2f}")

                if result.line_number:
                    report.append(f"Line: {result.line_number}")

                if result.error_message:
                    report.append(f"Message: {result.error_message}")

                if result.suggestions:
                    report.append("Suggestions:")
                    for suggestion in result.suggestions:
                        report.append(f"  • {suggestion}")

            report.append("-" * 40)

        return "\n".join(report)

    def save_json_report(self, results: List[Any], output_path: str) -> None:
        """
        Save results to JSON file

        Args:
            results: List of PredictionResult objects
            output_path: Path to save the JSON report
        """
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': len(results),
                'files_with_errors': sum(1 for r in results if r.error_type.value != 'no_error'),
                'error_rate': (sum(1 for r in results if r.error_type.value != 'no_error') / len(results)) * 100 if results else 0
            },
            'results': []
        }

        for result in results:
            result_dict = {
                'file_path': result.file_path,
                'language': result.language,
                'error_type': result.error_type.value,
                'confidence': result.confidence,
                'line_number': result.line_number,
                'error_message': result.error_message,
                'suggestions': result.suggestions or [],
                'all_errors': result.all_errors if hasattr(result, 'all_errors') and result.all_errors else None
            }
            report_data['results'].append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    def save_csv_report(self, results: List[Any], output_path: str) -> None:
        """
        Save results to CSV file

        Args:
            results: List of PredictionResult objects
            output_path: Path to save the CSV report
        """
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'file_path', 'language', 'error_type', 'confidence',
                'line_number', 'error_message', 'suggestions'
            ])

            # Data rows
            for result in results:
                suggestions_str = '; '.join(result.suggestions or [])
                writer.writerow([
                    result.file_path,
                    result.language,
                    result.error_type.value,
                    result.confidence,
                    result.line_number or '',
                    result.error_message or '',
                    suggestions_str
                ])

    def generate_html_report(self, results: List[Any], output_path: str) -> None:
        """
        Generate an HTML report with styling

        Args:
            results: List of PredictionResult objects
            output_path: Path to save the HTML report
        """
        html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Detection Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
        }
        .summary {
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
        }
        .summary-item {
            text-align: center;
        }
        .summary-number {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        .error-item {
            border: 1px solid #ddd;
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 8px;
            background-color: #fff;
        }
        .error-item.no-error {
            border-left: 5px solid #28a745;
        }
        .error-item.error {
            border-left: 5px solid #dc3545;
        }
        .error-item.warning {
            border-left: 5px solid #ffc107;
        }
        .file-path {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        .error-details {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        .detail-item {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .detail-label {
            font-weight: bold;
            color: #666;
            font-size: 0.9em;
        }
        .suggestions {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #2196f3;
        }
        .suggestions h4 {
            margin-top: 0;
            color: #1976d2;
        }
        .suggestions ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .error-type-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            color: white;
        }
        .no-error { background-color: #28a745; }
        .compile-error { background-color: #dc3545; }
        .runtime-error { background-color: #fd7e14; }
        .logic-error { background-color: #ffc107; color: #212529; }
        .syntax-error { background-color: #e83e8c; }
        .type-error { background-color: #6f42c1; }
        .memory-error { background-color: #20c997; }
        .timeout { background-color: #6c757d; }
        .unknown { background-color: #adb5bd; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Error Detection Report</h1>
            <p>Generated on {timestamp}</p>
        </div>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-number">{total_files}</div>
                <div>Total Files</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{error_files}</div>
                <div>Files with Errors</div>
            </div>
            <div class="summary-item">
                <div class="summary-number">{error_rate:.1f}%</div>
                <div>Error Rate</div>
            </div>
        </div>

        <div class="results">
            {results_html}
        </div>
    </div>
</body>
</html>'''

        # Generate results HTML
        results_html = []
        for result in results:
            error_class = "no-error" if result.error_type.value == "no_error" else "error"
            error_type_class = result.error_type.value.replace('_', '-')

            suggestions_html = ""
            if result.suggestions:
                suggestions_list = "\n".join([f"<li>{s}</li>" for s in result.suggestions])
                suggestions_html = f'''
                <div class="suggestions">
                    <h4>💡 Suggestions:</h4>
                    <ul>{suggestions_list}</ul>
                </div>'''

            line_info = f"Line {result.line_number}" if result.line_number else "N/A"
            confidence_percent = f"{result.confidence * 100:.1f}%"

            # Check if we have multiple errors
            multiple_errors_html = ""
            if hasattr(result, 'all_errors') and result.all_errors and len(result.all_errors) > 1:
                multiple_errors_html = f'<h3>Found {len(result.all_errors)} potential issues:</h3>'
                for i, error in enumerate(result.all_errors, 1):
                    error_type_val = error.get('type', 'unknown')
                    if hasattr(error_type_val, 'value'):
                        error_type_val = error_type_val.value
                    error_class_sub = error_type_val.replace('_', '-')
                    error_conf = error.get('confidence', result.confidence) * 100

                    error_sugg_html = ""
                    if error.get('suggestions'):
                        sugg_list = "\n".join([f"<li>{s}</li>" for s in error['suggestions']])
                        error_sugg_html = f'<ul>{sugg_list}</ul>'

                    multiple_errors_html += f'''
                    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                        <strong>Issue {i}:</strong><br>
                        Type: <span class="error-type-badge {error_class_sub}">{error_type_val}</span><br>
                        Confidence: {error_conf:.1f}%<br>
                        {f"Line: {error.get('line')}<br>" if error.get('line') else ""}
                        {f"Message: {error.get('message')}<br>" if error.get('message') else ""}
                        {error_sugg_html}
                    </div>'''

            result_html = f'''
            <div class="error-item {error_class}">
                <div class="file-path">📄 {result.file_path}</div>
                <div class="error-details">
                    <div class="detail-item">
                        <div class="detail-label">Language</div>
                        <div>{result.language}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Error Type</div>
                        <div><span class="error-type-badge {error_type_class}">{result.error_type.value}</span></div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Confidence</div>
                        <div>{confidence_percent}</div>
                    </div>
                </div>
                {multiple_errors_html if multiple_errors_html else f"{f'<div class=\"detail-item\"><strong>Line:</strong> {line_info}</div>' if result.line_number else ''}{f'<div class=\"detail-item\"><strong>Message:</strong> {result.error_message}</div>' if result.error_message else ''}{suggestions_html}"}
            </div>'''

            results_html.append(result_html)

        # Calculate summary statistics
        total_files = len(results)
        error_files = sum(1 for r in results if r.error_type.value != 'no_error')
        error_rate = (error_files / total_files) * 100 if total_files > 0 else 0

        # Fill template
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_files=total_files,
            error_files=error_files,
            error_rate=error_rate,
            results_html="\n".join(results_html)
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def create_visualization(self, results: List[Any], output_dir: str) -> None:
        """
        Create visualization charts for the results

        Args:
            results: List of PredictionResult objects
            output_dir: Directory to save visualization files
        """
        if not VISUALIZATION_AVAILABLE:
            print("Matplotlib/Seaborn not available. Skipping visualization.")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Error type distribution
        self._plot_error_distribution(results, output_dir / "error_distribution.png")

        # Confidence distribution
        self._plot_confidence_distribution(results, output_dir / "confidence_distribution.png")

        # Language breakdown
        self._plot_language_breakdown(results, output_dir / "language_breakdown.png")

    def _plot_error_distribution(self, results: List[Any], output_path: Path) -> None:
        """Plot error type distribution"""
        error_counts = {}
        for result in results:
            error_type = result.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(range(len(error_counts)))

        plt.pie(error_counts.values(), labels=error_counts.keys(),
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Error Type Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_distribution(self, results: List[Any], output_path: Path) -> None:
        """Plot confidence score distribution"""
        confidences = [result.confidence for result in results]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Number of Files', fontsize=12)
        plt.title('Confidence Score Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_language_breakdown(self, results: List[Any], output_path: Path) -> None:
        """Plot language breakdown"""
        language_counts = {}
        language_errors = {}

        for result in results:
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

            if result.error_type.value != 'no_error':
                language_errors[lang] = language_errors.get(lang, 0) + 1

        languages = list(language_counts.keys())
        total_counts = [language_counts[lang] for lang in languages]
        error_counts = [language_errors.get(lang, 0) for lang in languages]

        x = range(len(languages))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar([i - width/2 for i in x], total_counts, width, label='Total Files', alpha=0.8)
        plt.bar([i + width/2 for i in x], error_counts, width, label='Files with Errors', alpha=0.8)

        plt.xlabel('Programming Language', fontsize=12)
        plt.ylabel('Number of Files', fontsize=12)
        plt.title('Files by Programming Language', fontsize=16, fontweight='bold')
        plt.xticks(x, languages, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary_stats(self, results: List[Any]) -> None:
        """Print quick summary statistics"""
        if not results:
            print("No results to summarize.")
            return

        total_files = len(results)
        error_files = sum(1 for r in results if r.error_type.value != 'no_error')
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Count by error type
        error_types = {}
        for result in results:
            error_type = result.error_type.value
            error_types[error_type] = error_types.get(error_type, 0) + 1

        print("\n📊 QUICK SUMMARY")
        print("=" * 40)
        print(f"Total files: {total_files}")
        print(f"Files with errors: {error_files} ({(error_files/total_files)*100:.1f}%)")
        print(f"Average confidence: {avg_confidence:.2f}")
        print("\nError types:")
        for error_type, count in sorted(error_types.items()):
            print(f"  {error_type}: {count}")


# Example usage
if __name__ == "__main__":
    from error_predictor import ErrorDetectionModel, PredictionResult, ErrorType

    # Create some mock results for testing
    mock_results = [
        PredictionResult(
            file_path="test1.py",
            language="python",
            error_type=ErrorType.SYNTAX_ERROR,
            confidence=0.85,
            line_number=10,
            error_message="Missing colon after if statement",
            suggestions=["Add colon after if condition"]
        ),
        PredictionResult(
            file_path="test2.js",
            language="javascript",
            error_type=ErrorType.NO_ERROR,
            confidence=0.92
        )
    ]

    reporter = ErrorReporter()

    # Test console report
    print(reporter.format_console_report(mock_results))

    # Test summary stats
    reporter.print_summary_stats(mock_results)