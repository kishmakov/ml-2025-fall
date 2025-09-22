# Configuration file for notebook.

c = get_config()  #noqa

c.NotebookApp.token = ''      # Disable token auth
c.NotebookApp.password = ''   # Disable password auth
c.NotebookApp.open_browser = False

c.NotebookApp.notebook_dir = '/home/kishmakov/Repos/shad/ml-2025-fall/hw01'
