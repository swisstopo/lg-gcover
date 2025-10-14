"""
Configuration robuste de Rich Console pour Windows/PowerShell
Compatible Rich 14.0+
"""
import os
import sys
from rich.console import Console


def create_console() -> Console:
    """
    Créer une instance de Console Rich adaptée à l'environnement.

    Gère les cas suivants:
    - Windows PowerShell moderne (UTF-8 capable)
    - Windows cmd.exe legacy
    - Linux/macOS (UTF-8 natif)
    - Environnements CI/CD sans TTY
    """

    # Détection de l'environnement Windows
    is_windows = os.name == 'nt'

    # Détection si on est dans un vrai terminal ou un pipe/redirect
    is_interactive = sys.stdout.isatty()

    # Configuration de base
    console_kwargs = {}

    if is_windows:
        # Sur Windows, forcer UTF-8 dans l'environnement
        try:
            # Tenter de configurer UTF-8 pour stdout/stderr
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, OSError):
            pass

        # Détecter la version de la console Windows
        wt_session = os.environ.get('WT_SESSION')  # Windows Terminal
        ps_version = os.environ.get('PSModulePath')  # PowerShell
        vscode = os.environ.get('TERM_PROGRAM') == 'vscode'  # VS Code

        if wt_session or vscode or (ps_version and 'PowerShell\\7' in ps_version):
            # Console moderne - support UTF-8 complet
            console_kwargs = {
                'force_terminal': True if is_interactive else None,
                'legacy_windows': False,  # Permet les beaux caractères Unicode
            }
        else:
            # Console legacy (cmd.exe ou vieux PowerShell) - mode safe
            console_kwargs = {
                'force_terminal': True if is_interactive else None,
                'legacy_windows': True,  # Force ASCII box drawing
                'safe_box': True,  # Utilise +|-+ au lieu de ─│┌┐
            }
    else:
        # Linux/macOS - configuration standard
        console_kwargs = {
            'force_terminal': True if is_interactive else None,
        }

    return Console(**console_kwargs)


def create_safe_console() -> Console:
    """
    Version ultra-safe pour les environnements problématiques.
    Garantit l'affichage même sur les terminaux les plus basiques.
    """
    return Console(
        force_terminal=sys.stdout.isatty() or None,
        legacy_windows=os.name == 'nt',  # Active le mode ASCII sur Windows
        safe_box=True,  # Utilise des caractères ASCII pour les boîtes
        no_color=False,  # Garde les couleurs si possible
    )


def configure_windows_console():
    """
    Configure la console Windows pour un meilleur support UTF-8.
    À appeler au début de votre script sur Windows.
    """
    if os.name != 'nt':
        return

    try:
        # Essayer de passer en mode UTF-8 via les APIs Windows
        import ctypes
        kernel32 = ctypes.windll.kernel32

        # Code page 65001 = UTF-8
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)

    except Exception:
        # Si ça échoue, pas grave, on continue avec la config de base
        pass


# =============================================================================
# Solution SIMPLE et DIRECTE pour lg-gcover
# =============================================================================

# Option 1: Simple et fonctionnelle partout
def get_console_simple():
    """Configuration minimaliste qui marche partout"""
    if os.name == 'nt':
        return Console(legacy_windows=True, safe_box=True)
    return Console()


# Option 2: Optimisée selon l'environnement
def get_console_optimized():
    """Configuration optimisée selon l'environnement détecté"""
    if os.name == 'nt':
        # Windows Terminal ou VS Code = on peut utiliser Unicode
        if os.environ.get('WT_SESSION') or os.environ.get('TERM_PROGRAM') == 'vscode':
            return Console(legacy_windows=False)
        # Sinon mode safe
        return Console(legacy_windows=True, safe_box=True)
    return Console()


console = get_console_simple()