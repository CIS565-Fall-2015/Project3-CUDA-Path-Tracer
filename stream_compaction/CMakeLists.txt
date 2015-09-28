set(SOURCE_FILES
    "common.h"
    "common.cu"
    "stream_compaction.h"
    "stream_compaction.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
