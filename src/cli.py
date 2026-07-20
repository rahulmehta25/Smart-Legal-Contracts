import click
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.arbitration_detector import ArbitrationDetectionPipeline
from comparison.comparison_engine import ClauseComparisonEngine
from explainability.explainer import ArbitrationExplainer

console = Console()

@click.group()
def cli():
    """Arbitration Clause Detection CLI"""
    pass

@cli.command()
@click.argument('filepath', type=click.Path(exists=True))
@click.option('--explain', is_flag=True, help='Include explanation')
@click.option('--compare', is_flag=True, help='Compare with database')
@click.option('--output', type=click.Path(), help='Save results to file')
def detect(filepath, explain, compare, output):
    """Detect arbitration clause in document"""
    
    with console.status("[bold green]Analyzing document...") as status:
        pipeline = ArbitrationDetectionPipeline()
        result = pipeline.detect_arbitration_clause(filepath)
    
    if result:
        # Display results
        console.print(Panel(
            f"[bold green]✓ Arbitration Clause Detected[/bold green]\n"
            f"Confidence: {result.confidence:.1%}\n"
            f"Type: {result.clause_type}\n"
            f"Location: {result.location['section_title']}"
        ))
        
        # Show key provisions
        if result.key_provisions:
            table = Table(title="Key Provisions")
            table.add_column("Provision", style="cyan")
            for provision in result.key_provisions:
                table.add_row(provision)
            console.print(table)
        
        # Explanation
        if explain:
            explainer = ArbitrationExplainer(pipeline.bert_detector)
            explanation = explainer.explain_detection(result.full_text, result)
            
            console.print("\n[bold]Explanation:[/bold]")
            for step in explanation['decision_path']:
                console.print(f"  • {step}")
        
        # Comparison
        if compare:
            comparison_engine = ClauseComparisonEngine()
            comparison = comparison_engine.compare_clause(result.full_text)
            
            console.print("\n[bold]Similar Clauses:[/bold]")
            for clause in comparison['similar_clauses'][:3]:
                console.print(f"  • {clause['company']} ({clause['similarity']:.1%} similar)")
        
        # Save output
        if output:
            with open(output, 'w') as f:
                json.dump({
                    'detected': True,
                    'confidence': result.confidence,
                    'clause_type': result.clause_type,
                    'provisions': result.key_provisions,
                    'full_text': result.full_text
                }, f, indent=2)
            console.print(f"\n[green]Results saved to {output}[/green]")
    else:
        console.print("[red]No arbitration clause detected[/red]")

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def batch_process(directory):
    """Process multiple documents in a directory"""
    pipeline = ArbitrationDetectionPipeline()
    path = Path(directory)
    
    results = []
    for file_path in path.glob('**/*.pdf'):
        console.print(f"Processing {file_path.name}...")
        result = pipeline.detect_arbitration_clause(str(file_path))
        
        results.append({
            'file': file_path.name,
            'detected': result is not None,
            'confidence': result.confidence if result else 0.0
        })
    
    # Display summary
    table = Table(title="Batch Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Detected", style="green")
    table.add_column("Confidence", style="yellow")
    
    for r in results:
        table.add_row(
            r['file'],
            "✓" if r['detected'] else "✗",
            f"{r['confidence']:.1%}"
        )
    
    console.print(table)

@cli.command()
@click.argument('clause_text')
def compare(clause_text):
    """Compare a clause with the database"""
    comparison_engine = ClauseComparisonEngine()
    comparison = comparison_engine.compare_clause(clause_text)
    
    console.print("[bold]Comparison Results:[/bold]")
    console.print(f"Risk Assessment: {comparison['analysis']['risk_assessment']}")
    
    if comparison['similar_clauses']:
        table = Table(title="Similar Clauses")
        table.add_column("Company", style="cyan")
        table.add_column("Similarity", style="green")
        table.add_column("Risk Score", style="yellow")
        
        for clause in comparison['similar_clauses'][:5]:
            table.add_row(
                clause['company'],
                f"{clause['similarity']:.1%}",
                f"{clause['risk_score']:.2f}"
            )
        console.print(table)
    
    if comparison['analysis']['recommendations']:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in comparison['analysis']['recommendations']:
            console.print(f"  • {rec}")

if __name__ == '__main__':
    cli()
