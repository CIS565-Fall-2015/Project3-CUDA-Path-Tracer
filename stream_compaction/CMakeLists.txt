set(SOURCE_FILES
    "common.h"
    "shared.h"
    "shared.cu"
    )

cuda_add_library(stream_compaction
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
