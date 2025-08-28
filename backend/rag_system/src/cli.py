"""CLI tool for RAG-based arbitration detection system."""
import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
import time

from core.arbitration_detector import ArbitrationDetectionPipeline
from comparison.comparison_engine import ClauseComparisonEngine
from explainability.explainer import ArbitrationExplainer

console = Console()

@click.group()
@click.version_option(version='2.0.0')
def cli():
    """
    RAG Arbitration Detection CLI
    
    Advanced Legal-BERT based arbitration clause detection system.
    """
    pass

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--threshold', default=0.7, help='Detection confidence threshold (0-1)')
@click.option('--explain', is_flag=True, help='Include detailed explanation')
@click.option('--compare', is_flag=True, help='Compare with database')
@click.option('--output', type=click.Path(), help='Save results to file')
@click.option('--json', 'output_json', is_flag=True, help='Output in JSON format')
def detect(filepath, threshold, explain, compare, output, output_json):
    """Detect arbitration clause in a document."""
    
    with console.status(f"[bold green]Analyzing document: {filepath}...") as status:
        pipeline = ArbitrationDetectionPipeline()
        result = pipeline.detect_arbitration_clause(filepath)
    
    if result:
        if output_json:
            # JSON output
            output_data = result.to_dict()
            
            if compare:
                comparison_engine = ClauseComparisonEngine()
                comparison = comparison_engine.compare_clause(result.full_text)
                output_data['comparison'] = comparison
            
            if output:
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                console.print(f"[green]Results saved to {output}[/green]")
            else:
                console.print(json.dumps(output_data, indent=2))
        else:
            # Rich console output
            console.print(Panel(
                f"[bold green]✓ Arbitration Clause Detected[/bold green]\n"
                f"Confidence: [yellow]{result.confidence:.1%}[/yellow]\n"
                f"Type: [cyan]{result.clause_type}[/cyan]\n"
                f"Location: {result.location.get('section_title', 'Unknown')}\n"
                f"Detection Method: {result.detection_method}",
                title="Detection Result",
                border_style="green"
            ))
            
            # Show key provisions
            if result.key_provisions:
                table = Table(title="Key Provisions", show_header=False)
                table.add_column("Provision", style="cyan", no_wrap=False)
                for provision in result.key_provisions:
                    table.add_row(f"• {provision}")
                console.print(table)
            
            # Show summary
            console.print(Panel(
                result.summary,
                title="Clause Summary",
                border_style="blue"
            ))
            
            # Explanation
            if explain:
                console.print("\n[bold]Detailed Explanation:[/bold]")
                console.print(f"  • Detection confidence: {result.confidence:.2%}")
                console.print(f"  • Clause type: {result.clause_type}")
                console.print(f"  • Number of provisions: {len(result.key_provisions)}")
                console.print(f"  • Pages: {result.location.get('start_page', 'N/A')} - {result.location.get('end_page', 'N/A')}")
            
            # Comparison
            if compare:
                with console.status("[bold green]Comparing with database..."):
                    comparison_engine = ClauseComparisonEngine()
                    comparison = comparison_engine.compare_clause(result.full_text)
                
                if comparison['similar_clauses']:
                    console.print("\n[bold]Similar Clauses Found:[/bold]")
                    similar_table = Table(show_header=True, header_style="bold magenta")
                    similar_table.add_column("Company", style="cyan")
                    similar_table.add_column("Industry", style="green")
                    similar_table.add_column("Similarity", justify="right", style="yellow")
                    similar_table.add_column("Risk", justify="right", style="red")
                    
                    for clause in comparison['similar_clauses'][:5]:
                        similar_table.add_row(
                            clause['company'],
                            clause['industry'],
                            f"{clause['similarity']:.1%}",
                            f"{clause.get('risk_score', 0):.1%}"
                        )
                    console.print(similar_table)
                
                # Show recommendations
                if comparison.get('analysis', {}).get('recommendations'):
                    console.print("\n[bold]Recommendations:[/bold]")
                    for rec in comparison['analysis']['recommendations']:
                        console.print(f"  [yellow]→[/yellow] {rec}")
            
            # Save output
            if output:
                with open(output, 'w') as f:
                    f.write(f"Arbitration Clause Detection Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"File: {filepath}\n")
                    f.write(f"Detected: Yes\n")
                    f.write(f"Confidence: {result.confidence:.1%}\n")
                    f.write(f"Type: {result.clause_type}\n")
                    f.write(f"\nKey Provisions:\n")
                    for provision in result.key_provisions:
                        f.write(f"  - {provision}\n")
                    f.write(f"\nSummary:\n{result.summary}\n")
                console.print(f"\n[green]Results saved to {output}[/green]")
    else:
        console.print(Panel(
            "[red]✗ No arbitration clause detected[/red]",
            title="Detection Result",
            border_style="red"
        ))

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--pattern', default='*.pdf', help='File pattern to match')
@click.option('--recursive', is_flag=True, help='Search recursively')
@click.option('--output', type=click.Path(), help='Save results to CSV file')
def batch(directory, pattern, recursive, output):
    """Process multiple documents in a directory."""
    
    pipeline = ArbitrationDetectionPipeline()
    path = Path(directory)
    
    # Find files
    if recursive:
        files = list(path.rglob(pattern))
    else:
        files = list(path.glob(pattern))
    
    if not files:
        console.print(f"[red]No files matching '{pattern}' found in {directory}[/red]")
        return
    
    console.print(f"Found [cyan]{len(files)}[/cyan] files to process")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=len(files))
        
        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}...")
            
            try:
                result = pipeline.detect_arbitration_clause(str(file_path))
                
                results.append({
                    'file': file_path.name,
                    'path': str(file_path),
                    'detected': result is not None,
                    'confidence': result.confidence if result else 0.0,
                    'type': result.clause_type if result else 'N/A',
                    'provisions': len(result.key_provisions) if result else 0
                })
            except Exception as e:
                console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                results.append({
                    'file': file_path.name,
                    'path': str(file_path),
                    'detected': False,
                    'error': str(e)
                })
            
            progress.advance(task)
    
    # Display summary
    detected_count = sum(1 for r in results if r['detected'])
    
    console.print(Panel(
        f"Processed: [cyan]{len(results)}[/cyan] files\n"
        f"Detected: [green]{detected_count}[/green] with arbitration clauses\n"
        f"Not detected: [red]{len(results) - detected_count}[/red] without arbitration clauses",
        title="Batch Processing Summary",
        border_style="blue"
    ))
    
    # Display results table
    if results:
        table = Table(title="Processing Results", show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan", no_wrap=True)
        table.add_column("Detected", justify="center")
        table.add_column("Confidence", justify="right", style="yellow")
        table.add_column("Type", style="green")
        table.add_column("Provisions", justify="right")
        
        for r in results:
            detected_symbol = "✓" if r['detected'] else "✗"
            detected_style = "green" if r['detected'] else "red"
            
            table.add_row(
                r['file'][:50],  # Truncate long filenames
                f"[{detected_style}]{detected_symbol}[/{detected_style}]",
                f"{r.get('confidence', 0):.1%}" if r['detected'] else "-",
                r.get('type', '-'),
                str(r.get('provisions', '-'))
            )
        
        console.print(table)
    
    # Save to CSV if requested
    if output:
        import csv
        with open(output, 'w', newline='') as csvfile:
            fieldnames = ['file', 'path', 'detected', 'confidence', 'type', 'provisions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, '') for k in fieldnames})
        console.print(f"\n[green]Results saved to {output}[/green]")

@cli.command()
@click.argument('clause_text')
@click.option('--top-k', default=10, help='Number of similar clauses to return')
def compare(clause_text, top_k):
    """Compare a clause with the database."""
    
    with console.status("[bold green]Comparing clause with database..."):
        comparison_engine = ClauseComparisonEngine()
        comparison = comparison_engine.compare_clause(clause_text, top_k)
    
    if comparison['similar_clauses']:
        # Display similar clauses
        table = Table(title="Similar Clauses", show_header=True, header_style="bold magenta")
        table.add_column("Company", style="cyan")
        table.add_column("Industry", style="green")
        table.add_column("Doc Type", style="blue")
        table.add_column("Similarity", justify="right", style="yellow")
        table.add_column("Risk", justify="right", style="red")
        
        for clause in comparison['similar_clauses']:
            table.add_row(
                clause['company'],
                clause['industry'],
                clause['document_type'],
                f"{clause['similarity']:.1%}",
                f"{clause.get('risk_score', 0):.1%}"
            )
        console.print(table)
        
        # Display analysis
        analysis = comparison.get('analysis', {})
        
        if analysis.get('risk_assessment'):
            console.print(Panel(
                f"Risk Assessment: {analysis['risk_assessment']}",
                border_style="yellow"
            ))
        
        if analysis.get('recommendations'):
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in analysis['recommendations']:
                console.print(f"  [yellow]→[/yellow] {rec}")
        
        # Display statistics
        stats = comparison.get('statistics', {})
        if stats:
            console.print("\n[bold]Statistics:[/bold]")
            console.print(f"  Total similar clauses: {stats.get('total_similar_clauses', 0)}")
            console.print(f"  Average similarity: {stats.get('average_similarity', 0):.1%}")
            console.print(f"  Average risk: {stats.get('average_risk', 0):.1%}")
            console.print(f"  Average enforceability: {stats.get('average_enforceability', 0):.1%}")
    else:
        console.print("[yellow]No similar clauses found in database[/yellow]")

@cli.command()
def stats():
    """Display database statistics."""
    
    with console.status("[bold green]Loading database statistics..."):
        comparison_engine = ClauseComparisonEngine()
        stats = comparison_engine.get_database_stats()
    
    if 'error' in stats:
        console.print(f"[red]Error: {stats['error']}[/red]")
        return
    
    # Display database stats
    db_stats = stats.get('database', {})
    console.print(Panel(
        f"Total Clauses: [cyan]{db_stats.get('total_clauses', 0)}[/cyan]\n"
        f"Unique Companies: [green]{db_stats.get('unique_companies', 0)}[/green]\n"
        f"Unique Industries: [blue]{db_stats.get('unique_industries', 0)}[/blue]\n"
        f"Unique Jurisdictions: [yellow]{db_stats.get('unique_jurisdictions', 0)}[/yellow]\n"
        f"Average Risk Score: [red]{db_stats.get('average_risk_score', 0):.1%}[/red]\n"
        f"Average Enforceability: [magenta]{db_stats.get('average_enforceability', 0):.1%}[/magenta]",
        title="Database Statistics",
        border_style="blue"
    ))
    
    # Display vector store stats
    vector_stats = stats.get('vector_store', {})
    if vector_stats:
        console.print(Panel(
            f"Total Vectors: [cyan]{vector_stats.get('total_vectors', 0)}[/cyan]\n"
            f"Dimension: [green]{vector_stats.get('dimension', 0)}[/green]\n"
            f"Mapped IDs: [blue]{vector_stats.get('mapped_ids', 0)}[/blue]",
            title="Vector Store Statistics",
            border_style="green"
        ))

@cli.command()
@click.argument('json_file', type=click.Path(exists=True))
def import_db(json_file):
    """Import clauses from JSON file to database."""
    
    try:
        with open(json_file, 'r') as f:
            clauses = json.load(f)
        
        if not isinstance(clauses, list):
            console.print("[red]Error: JSON file must contain a list of clauses[/red]")
            return
        
        console.print(f"Importing [cyan]{len(clauses)}[/cyan] clauses...")
        
        with console.status("[bold green]Importing clauses to database..."):
            comparison_engine = ClauseComparisonEngine()
            result = comparison_engine.bulk_import_clauses(clauses)
        
        console.print(Panel(
            f"Total: [cyan]{result['total']}[/cyan]\n"
            f"Success: [green]{result['success']}[/green]\n"
            f"Errors: [red]{result['errors']}[/red]",
            title="Import Results",
            border_style="green" if result['errors'] == 0 else "yellow"
        ))
        
        if result.get('error_details'):
            console.print("\n[bold red]Errors:[/bold red]")
            for error in result['error_details'][:5]:
                console.print(f"  • {error['clause']}: {error['error']}")
    
    except Exception as e:
        console.print(f"[red]Error importing clauses: {e}[/red]")

@cli.command()
def serve():
    """Start the API server."""
    console.print("[bold green]Starting RAG Arbitration Detection API server...[/bold green]")
    console.print("Access the API documentation at: [cyan]http://localhost:8000/api/docs[/cyan]")
    
    import uvicorn
    from api.main import app
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")

if __name__ == '__main__':
    cli()