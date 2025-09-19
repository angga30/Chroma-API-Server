#!/usr/bin/env python3
import click
import json
from typing import Optional, List
from service.chromadb import chroma_service
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

@click.group()
def cli():
    """CLI tool untuk manajemen ChromaDB"""
    pass

@cli.command()
@click.argument('collection_name')
@click.option('--limit', '-l', default=10, help='Jumlah dokumen yang ditampilkan')
@click.option('--where', '-w', help='Filter query dalam format JSON (contoh: \'{"category": "technology"}\')')
def list_documents(collection_name: str, limit: int, where: Optional[str]):
    """Menampilkan dokumen dalam collection"""
    try:
        collection = chroma_service.get_or_create_collection(collection_name)
        where_filter = json.loads(where) if where else None
        
        # Query documents
        results = collection.query(
            query_texts=[""],  # Empty query to get all documents
            n_results=limit,
            where=where_filter
        )
        
        if not results['ids'][0]:
            console.print(f"[yellow]Tidak ada dokumen dalam collection '{collection_name}'[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Documents in {collection_name}")
        table.add_column("ID", style="cyan")
        table.add_column("Content", style="green")
        table.add_column("Metadata", style="yellow")
        
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            content = results['documents'][0][i][:100] + "..." if len(results['documents'][0][i]) > 100 else results['documents'][0][i]
            metadata = json.dumps(results['metadatas'][0][i], indent=2)
            table.add_row(doc_id, content, metadata)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
def list_collections():
    """Menampilkan semua collections"""
    try:
        collections = chroma_service.list_collections()
        
        if not collections:
            console.print("[yellow]Tidak ada collections yang tersedia[/yellow]")
            return
        
        table = Table(title="Available Collections")
        table.add_column("Collection Name", style="cyan")
        
        for collection in collections:
            table.add_row(collection)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.argument('collection_name')
@click.argument('query')
@click.option('--n-results', '-n', default=5, help='Jumlah hasil yang ditampilkan')
@click.option('--threshold', '-t', default=0.5, help='Threshold similarity (0-1)')
def search(collection_name: str, query: str, n_results: int, threshold: float):
    """Mencari dokumen berdasarkan similarity"""
    try:
        results = chroma_service.search_similarity(
            collection_name=collection_name,
            query=query,
            n_results=n_results,
            threshold=threshold
        )
        
        if not results['ids']:
            console.print(f"[yellow]Tidak ditemukan dokumen yang sesuai dengan query '{query}'[/yellow]")
            return
        
        table = Table(title=f"Search Results for '{query}'")
        table.add_column("ID", style="cyan")
        table.add_column("Content", style="green")
        table.add_column("Similarity", style="yellow")
        table.add_column("Metadata", style="magenta")
        
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            content = results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i]
            similarity = f"{(1 - results['distances'][i]):.2f}"
            metadata = json.dumps(results['metadatas'][i], indent=2)
            table.add_row(doc_id, content, similarity, metadata)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.argument('collection_name')
@click.argument('doc_ids', nargs=-1)
def delete(collection_name: str, doc_ids: List[str]):
    """Menghapus dokumen berdasarkan ID"""
    try:
        if not doc_ids:
            console.print("[red]Error: Harap berikan minimal satu document ID[/red]")
            return
        
        chroma_service.delete_documents(collection_name, list(doc_ids))
        console.print(f"[green]Berhasil menghapus {len(doc_ids)} dokumen dari collection '{collection_name}'[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.argument('collection_name')
def delete_collection(collection_name: str):
    """Menghapus collection"""
    try:
        chroma_service.client.delete_collection(collection_name)
        console.print(f"[green]Berhasil menghapus collection '{collection_name}'[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")

if __name__ == '__main__':
    cli()