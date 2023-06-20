import os
import sys

script = str(sys.argv[1])
cpu_id = int(sys.argv[2])
seg_id = int(sys.argv[3])
os.system('taskset -c %d %s.py %d %d' % (cpu_id, script cpu_id, seg_id))