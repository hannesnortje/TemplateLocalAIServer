import sys
import click
import os
from . import __version__
from .__main__ import main as server_main

@click.group()
@click.version_option(version=__version__)
def cli():
    """Local AI Server - Run language models locally"""
    pass

@cli.command()
@click.option('--http-port', type=int, help='HTTP port (default: 5000)')
@click.option('--https-port', type=int, help='HTTPS port (default: 5443)')
@click.option('--models-dir', type=click.Path(), help='Models directory')
@click.option('--vector-db', type=click.Choice(['qdrant', 'chroma']), help='Vector database to use')
def start(http_port, https_port, models_dir, vector_db):
    """Start the Local AI Server"""
    if http_port:
        os.environ['HTTP_PORT'] = str(http_port)
    if https_port:
        os.environ['HTTPS_PORT'] = str(https_port)
    if models_dir:
        os.environ['MODELS_DIR'] = str(models_dir)
    if vector_db:
        os.environ['VECTOR_DB_TYPE'] = vector_db
        click.echo(f"Using vector database: {vector_db}")
    
    sys.exit(server_main())

@cli.command()
def models():
    """List available and installed models"""
    from .model_manager import model_manager
    installed = model_manager.list_models()
    available = model_manager.get_status()
    
    click.echo("Installed models:")
    for model in installed:
        click.echo(f"- {model['id']} ({model['type']})")
    
    click.echo("\nAvailable models:")
    for name, info in available.items():
        click.echo(f"- {name} ({'loaded' if info.loaded else 'not loaded'})")

if __name__ == '__main__':
    cli()
