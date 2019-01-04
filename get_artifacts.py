
import paperspace

files = paperspace.jobs.artifactsGet({'jobId': 'jsrsmgaq3iqbbj', 'dest': '~/dk_ps/deepknockoff_grf/ps_files'}, no_logging=True)
paperspace.print_json_pretty(files)