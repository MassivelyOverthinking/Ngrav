#==================================================================================================================
# IMPORTS
#==================================================================================================================

from uuid import uuid4
from datetime import UTC, datetime
from time import perf_counter_ns

#==================================================================================================================
# UTILITY FUNCTIONS
#==================================================================================================================

def get_initial_execution_metadat(self):
    execution_id = uuid4()
    started_at = datetime.now(UTC)
    started_ns = perf_counter_ns()

    return execution_id, started_at, started_ns