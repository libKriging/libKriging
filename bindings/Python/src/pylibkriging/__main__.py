from . import __version__, __build_type__

note = ''
if __build_type__ != 'Release':
    note = f" [Warning: {__build_type__} build]"

print(f"""
pylibkriging {__version__} is ready!${note}
Use 'import pylibkriging'
""")
